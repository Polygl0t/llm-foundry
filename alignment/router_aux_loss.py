"""MoE router auxiliary (load-balancing) loss for the alignment trainers.

Copied from `llm-foundry/distributed/model_setup.py`.
TODO: Implement this functionality for both `alignment` and `distributed` in
a shared module (i.e., `/llm-foundry/shared`).
"""

import sys

import torch

# Rows of routing logits handled per softmax chunk by the efficient aux loss.
AUX_LOSS_ROW_CHUNK = 1 << 16

# transformers' own `load_balancing_loss_func`, kept so the efficient version can
# delegate unusual input layouts back to it.
_ORIGINAL_LOAD_BALANCING_LOSS = None

# Class-name fragments that identify a router ("gate") module. Matches
# `Qwen3_5MoeTopKRouter`, `Qwen3MoeTopKRouter`, `DeepseekV3TopKRouter`,
# `MixtralBlockSparseTop2MLP`-style gates (`...MoEGate`), etc.
ROUTER_CLASS_PATTERNS = ("Router", "MoEGate")


class _EfficientLoadBalancingLoss(torch.autograd.Function):
    """
    Switch-Transformer load-balancing loss without the `one_hot` blow-up.

    The loss only needs two statistics, both additive over layers and rows:

        assignments[e] = number of top-k slots that picked expert `e`
        prob_sum[e]    = summed routing probability of expert `e`

        loss = num_experts * sum_e (assignments[e]/M) * (prob_sum[e]/M)

    with `M` the number of routing rows (optionally mask-weighted). It is
    therefore computed one layer at a time, in row chunks, and never
    concatenates the per-layer logits. `assignments` is piecewise constant in the
    logits (it comes from an argmax), so the backward pass has a closed form and
    nothing has to be retained for it:

        d loss / d logit[n, j] = (num_experts * m[n] / M) * p[n, j] * (f[j] - <f, p[n]>)

    with `f[e] = assignments[e]/M`, `p[n]` the routing distribution of row `n`
    and `m[n]` its weight (1 without an attention mask). Verified against the
    reference: loss to 4e-8 relative, gradient to 2e-10 absolute (fp32); through
    Liger's `lce_forward` in bf16 the total loss matches to 7e-8 relative and
    `aux_loss` bit-for-bit.

    NOTE: does not support double backward (`create_graph=True`).
    """

    @staticmethod
    def forward(ctx, num_experts, top_k, row_weights, *layer_logits):
        # NOTE: the per-layer logits are passed as VARARGS, not as one tuple. A tuple
        # input to `Function.apply` is not tracked by autograd, which silently produced
        # a non-differentiable loss.
        device = layer_logits[0].device
        assignments = torch.zeros(num_experts, device=device, dtype=torch.float32)
        prob_sum = torch.zeros(num_experts, device=device, dtype=torch.float32)

        with torch.no_grad():
            for logits in layer_logits:
                for start in range(0, logits.shape[0], AUX_LOSS_ROW_CHUNK):
                    stop = min(start + AUX_LOSS_ROW_CHUNK, logits.shape[0])
                    probs = torch.softmax(logits[start:stop].float(), dim=-1)
                    selected = torch.topk(probs, top_k, dim=-1).indices.reshape(-1)
                    if row_weights is None:
                        assignments += torch.bincount(selected, minlength=num_experts).to(
                            torch.float32
                        )
                        prob_sum += probs.sum(dim=0)
                    else:
                        weights = row_weights[start:stop].float()
                        assignments += torch.bincount(
                            selected,
                            weights=weights.unsqueeze(-1).expand(-1, top_k).reshape(-1),
                            minlength=num_experts,
                        )
                        prob_sum += (probs * weights.unsqueeze(-1)).sum(dim=0)
                    del probs, selected

            if row_weights is None:
                total = float(sum(logits.shape[0] for logits in layer_logits))
            else:
                # The same mask applies to every layer.
                total = float(row_weights.sum()) * len(layer_logits)

        fractions = assignments / total
        probabilities = prob_sum / total
        loss = (fractions * probabilities).sum() * num_experts

        ctx.save_for_backward(*layer_logits)
        ctx.fractions = fractions.detach()
        ctx.num_experts = num_experts
        ctx.total = total
        ctx.row_weights = None if row_weights is None else row_weights.detach()
        ctx.needs_grad = tuple(logits.requires_grad for logits in layer_logits)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        fractions = ctx.fractions
        scale = grad_output.to(torch.float32) * (ctx.num_experts / ctx.total)
        grads = []
        for index, logits in enumerate(ctx.saved_tensors):
            if not ctx.needs_grad[index]:
                grads.append(None)
                continue
            grad = torch.empty_like(logits)
            for start in range(0, logits.shape[0], AUX_LOSS_ROW_CHUNK):
                stop = min(start + AUX_LOSS_ROW_CHUNK, logits.shape[0])
                probs = torch.softmax(logits[start:stop].float(), dim=-1)
                expected = (probs * fractions).sum(dim=-1, keepdim=True)
                update = probs * (fractions - expected) * scale
                if ctx.row_weights is not None:
                    update = update * ctx.row_weights[start:stop].float().unsqueeze(-1)
                grad[start:stop] = update.to(logits.dtype)
            grads.append(grad)
        # One `None` each for num_experts, top_k and row_weights.
        return (None, None, None, *grads)


def efficient_load_balancing_loss(gate_logits, num_experts=None, top_k=2, attention_mask=None):
    """Drop-in replacement for `transformers...modeling_*.load_balancing_loss_func`."""
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if num_experts is None:
        num_experts = gate_logits[0].shape[-1]
    num_experts = int(num_experts)
    top_k = int(top_k)

    # transformers only supports 2-D `(batch * seq, num_experts)` logits here (its
    # attention-mask branch derives everything from that shape).
    if any(logits.dim() != 2 for logits in gate_logits):
        if _ORIGINAL_LOAD_BALANCING_LOSS is None:
            raise RuntimeError(
                "Expected `router_logits` to be 2-D (batch*seq, num_experts) but got shape "
                f"{tuple(gate_logits[0].shape)} and no reference implementation is available."
            )
        return _ORIGINAL_LOAD_BALANCING_LOSS(gate_logits, num_experts, top_k, attention_mask)

    row_weights = None
    if attention_mask is not None:
        if attention_mask.dim() != 2 or attention_mask.numel() != gate_logits[0].shape[0]:
            if _ORIGINAL_LOAD_BALANCING_LOSS is None:
                raise RuntimeError(
                    "`attention_mask` does not match the router logits "
                    f"(shape {tuple(attention_mask.shape)}, {gate_logits[0].shape[0]} rows) and no "
                    "reference implementation is available."
                )
            return _ORIGINAL_LOAD_BALANCING_LOSS(gate_logits, num_experts, top_k, attention_mask)
        row_weights = attention_mask.reshape(-1)

    return _EfficientLoadBalancingLoss.apply(num_experts, top_k, row_weights, *gate_logits)


def _reference_function(model):
    """The model's own `load_balancing_loss_func`, or None for dense models."""
    module = sys.modules.get(type(model).__module__)
    if module is None:
        return None
    binding = vars(module).get("load_balancing_loss_func")
    if binding is not None:
        return binding
    # Lazy modules expose the name through `__getattr__` instead of `__dict__`.
    try:
        return getattr(module, "load_balancing_loss_func", None)
    except Exception:  # noqa: BLE001 - harmless if a lazy module cannot resolve it
        return None


def install_efficient_load_balancing_loss(reference, logger=None):
    """
    Bind `efficient_load_balancing_loss` everywhere `reference` was bound.

    Returns:
        List of patched module names.
    """
    global _ORIGINAL_LOAD_BALANCING_LOSS

    if _ORIGINAL_LOAD_BALANCING_LOSS is None:
        _ORIGINAL_LOAD_BALANCING_LOSS = reference

    patched = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            bindings = vars(module)
        except TypeError:  # modules with a custom __dict__ type
            continue
        if bindings.get("load_balancing_loss_func") is reference:
            bindings["load_balancing_loss_func"] = efficient_load_balancing_loss
            patched.append(name)

    if logger is not None:
        logger.info(
            "Router aux loss: installed the memory-efficient `load_balancing_loss_func` in "
            f"{len(patched)} module(s): {', '.join(sorted(patched)) or 'none'}"
        )
    return patched


def _moe_configs(model):
    """Config objects of `model` that actually describe a Mixture-of-Experts model."""
    config = getattr(model, "config", None)
    if config is None:
        return []
    found = []
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is None or any(candidate is seen for seen in found):
            continue
        num_experts = (
            getattr(candidate, "num_experts", None)
            or getattr(candidate, "num_local_experts", None)
            or 0
        )
        if num_experts > 1:
            found.append(candidate)
    return found


def estimate_aux_loss_memory(model, max_length):
    """GiB transformers' implementation would allocate at this shape, or None."""
    configs = _moe_configs(model)
    if not configs:
        return None
    config = configs[0]
    num_experts = int(
        getattr(config, "num_experts", None) or getattr(config, "num_local_experts", 0) or 0
    )
    top_k = int(getattr(config, "num_experts_per_tok", 0) or 0)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if not (num_experts and top_k and num_layers):
        return None
    gib = 1024**3
    return (
        num_layers * max_length * top_k * num_experts * 8 / gib,
        num_layers * max_length * num_experts * 4 / gib,
        num_layers * max_length * num_experts * 2 / gib,
    )


def find_router_modules(model):
    """`(name, module)` pairs for the expert-selection gates of `model`."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if any(pattern in type(module).__name__ for pattern in ROUTER_CLASS_PATTERNS)
    ]


def freeze_router_modules(model, logger=None):
    """Stop gradient flow into every router/gate parameter.

    Note: This only freezes the router parameters. It does NOT
    preserve the routing *behaviour*.
    """
    frozen = []
    for module_name, module in find_router_modules(model):
        for param_name, param in module.named_parameters(recurse=True):
            param.requires_grad_(False)
            frozen.append(f"{module_name}.{param_name}")
    if logger is not None:
        if frozen:
            logger.info(
                f"Routers FROZEN ({len(frozen)} tensors): {', '.join(frozen)}. The routing "
                "function is preserved bit-for-bit; assignments can still shift because the "
                "hidden states keep training."
            )
        else:
            logger.warning(
                "freeze_router was requested but no router module was found "
                f"(looked for classes matching {ROUTER_CLASS_PATTERNS}); nothing was frozen."
            )
    return frozen


def configure_router_aux_loss(model, coef, logger=None, freeze_router=False):
    """Apply the router policy for this run and report what was done.

    Args:
        model: the model `Trainer` will train (policy model for DPO).
        coef: the requested auxiliary-loss coefficient. `None` means "inherit
            whatever the model config says" (the trainers that do not expose the
            knob must not have their config rewritten); `0`/`0.0` disables it.
        logger: logger for the decision (rank 0 only, in practice).
        freeze_router: also freeze the router/gate parameters.

    Returns:
        Dict with `enabled`, `coefficient`, `frozen`, `patched_modules`, `memory_gib`.

    Raises:
        RuntimeError: when `coef > 0` but the model exposes no
            `load_balancing_loss_func` to replace.
    """
    requested = coef
    coef = 0.0 if coef is None else float(coef)
    moe_configs = _moe_configs(model)
    status = {
        "enabled": None if requested is None else False,
        "coefficient": coef,
        "frozen": [],
        "patched_modules": [],
        "memory_gib": None,
        "is_moe": bool(moe_configs),
    }

    if freeze_router:
        status["frozen"] = freeze_router_modules(model, logger)

    if requested is None:
        # No opinion: leave `output_router_logits` / `router_aux_loss_coef` exactly
        # as the checkpoint left them (e.g. the GRPO and reward trainers).
        if logger is not None and moe_configs:
            logger.info(
                "Router aux loss: not configured for this trainer, inheriting the model "
                f"config (output_router_logits={getattr(moe_configs[0], 'output_router_logits', None)})."
            )
        return status

    if not moe_configs:
        if coef and logger is not None:
            logger.warning(
                f"router_aux_loss_coef={coef} was requested but the model has no "
                "Mixture-of-Experts layers; ignoring it."
            )
        return status

    if coef <= 0.0:
        for config in moe_configs:
            config.output_router_logits = False
            config.router_aux_loss_coef = 0.0
        _set_model_coef(model, 0.0)
        if logger is not None:
            logger.info(
                "Router aux loss disabled (coefficient 0). Routers stay trainable and are "
                "driven by the LM loss alone -- the configuration with the worst measured "
                "expert balance; pass --router_aux_loss_coef 0.001 to keep them guarded."
            )
        return status

    reference = _reference_function(model)
    if reference is None:
        raise RuntimeError(
            f"router_aux_loss_coef={coef} but {type(model).__name__} exposes no "
            "`load_balancing_loss_func`, so transformers' memory-hungry implementation cannot "
            "be replaced. Refusing to start: at long sequence lengths it needs ~1 MiB per "
            "token (80 GiB for one 131k-token sequence). Set --router_aux_loss_coef 0.0 to "
            "train without the auxiliary loss."
        )

    status["patched_modules"] = install_efficient_load_balancing_loss(reference, logger)

    for config in moe_configs:
        config.output_router_logits = True
        config.router_aux_loss_coef = coef
    # The forwards read the value cached on the model at __init__
    # (`model.router_aux_loss_coef`), not the config, so TRL setting the config
    # after loading is not enough.
    _set_model_coef(model, coef)

    status["enabled"] = True
    status["memory_gib"] = estimate_aux_loss_memory(model, _max_length_of(model) or 0)

    if logger is not None:
        config = moe_configs[0]
        num_experts = int(
            getattr(config, "num_experts", None) or getattr(config, "num_local_experts", 0) or 0
        )
        top_k = int(getattr(config, "num_experts_per_tok", 0) or 0)
        logger.info(
            f"Router aux loss ENABLED (coefficient={coef:g}) over {num_experts} experts, "
            f"top-{top_k}, using the memory-efficient implementation "
            f"({len(status['patched_modules'])} module(s) patched). Target aux_loss ~= top_k "
            f"= {top_k:g}; it approaches {num_experts} if routing collapses."
        )
        if status["frozen"]:
            logger.info(
                "Routers are frozen and the auxiliary loss is still enabled: the term no longer "
                "moves the routers, but it still reaches the experts/attention through the "
                "router's input. That is a deliberate 'balanced usage under a fixed router' "
                "policy; it costs the retained router logits."
            )
    return status


def _set_model_coef(model, coef):
    """Push `coef` into every object that caches `router_aux_loss_coef`."""
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "router_aux_loss_coef"):
            candidate.router_aux_loss_coef = coef


def _max_length_of(model):
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        config = getattr(candidate, "config", None)
        if config is None:
            continue
        for cfg in (config, getattr(config, "text_config", None)):
            max_length = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
            if max_length:
                return int(max_length)
    return None


def _local(tensor):
    """The local shard of a (possibly DTensor) tensor."""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def snapshot_router(model):
    """Detached copy of the router parameters, for `router_drift()`."""
    return {
        name: _local(param.detach()).float().clone()
        for name, param in model.named_parameters()
        if any(pattern in name for pattern in ROUTER_CLASS_PATTERNS)
        or "router" in name.lower()
        or ".gate.weight" in name
    }


def router_drift(model, snapshot):
    """Relative change of each router tensor since `snapshot` (max over tensors)."""
    if not snapshot:
        return None
    drifts = []
    for name, param in model.named_parameters():
        baseline = snapshot.get(name)
        if baseline is None:
            continue
        current = _local(param.detach()).float()
        if current.shape != baseline.shape:
            continue
        reference_norm = baseline.norm()
        if reference_norm == 0:
            continue
        drifts.append(float((current - baseline).norm() / reference_norm))
    return max(drifts) if drifts else None


def attach_aux_loss_logging(trainer, logger, drift_every=1):
    """Add `aux_loss` and `router_drift` to the trainer's own log lines."""
    model = getattr(trainer, "model", None)
    if model is None:
        return None

    state = {"total": 0.0, "count": 0, "drift": None}
    snapshot = snapshot_router(model)
    log_calls = {"n": 0}

    original_forward = model.forward

    def forward_with_aux(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        aux = getattr(outputs, "aux_loss", None)
        if isinstance(aux, torch.Tensor):
            state["total"] += float(aux.detach())
            state["count"] += 1
        return outputs

    model.forward = forward_with_aux

    original_log = trainer.log

    def log_with_aux(logs, *args, **kwargs):
        if getattr(model, "training", True) and state["count"]:
            logs["aux_loss"] = round(state["total"] / state["count"], 6)
            state["total"], state["count"] = 0.0, 0
        log_calls["n"] += 1
        if snapshot and drift_every and log_calls["n"] % drift_every == 0:
            drift = router_drift(model, snapshot)
            if drift is not None:
                logs["router_drift"] = round(drift, 6)
                state["drift"] = drift
        return original_log(logs, *args, **kwargs)

    trainer.log = log_with_aux
    if logger is not None and snapshot:
        logger.info(
            f"Router drift tracking on {len(snapshot)} tensor(s); `aux_loss` and `router_drift` "
            "(relative change since training started) will appear in the logs."
        )
    return state
