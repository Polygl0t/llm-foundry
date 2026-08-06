"""
Numerical debugging instrumentation for the distributed trainers.

Opt-in, zero-cost-when-disabled probes that answer the question "which tensor in
which module goes non-finite first, and does it happen in the forward pass, the
backward pass, or the optimizer update?".

The instrumentation never changes model outputs: every probe detaches the tensor
it inspects and returns nothing. Under DDP/FSDP only rank 0 computes and prints
(set `LLMF_DEBUG_ALL_RANKS=1` to compute on every rank, prefixed by rank id).

    LLMF_DEBUG=1                  master switch
    LLMF_DEBUG_VERBOSE_STEPS=2    dump full stats for the first N iterations
    LLMF_DEBUG_EVERY=0            also dump full stats every N iterations (0=off)
    LLMF_DEBUG_PARAM_EVERY=50     dump parameter stats every N iterations
    LLMF_DEBUG_MAX_ABS=1e4        abs_max above this is flagged EXPLODING
    LLMF_DEBUG_SMALL_ABS=1e-8     abs_max below this is flagged VANISHING
    LLMF_DEBUG_MAX_ANOMALIES=40   stop reporting after N anomalies
    LLMF_DEBUG_STOP_ON_NAN=0      raise RuntimeError on the first non-finite tensor
    LLMF_DEBUG_ALL_MODULES=0      also probe attention / MLP / MoE modules
    LLMF_DEBUG_ALL_RANKS=0        compute on every rank instead of rank 0 only
    LLMF_DEBUG_FILE=<path>        mirror the log to a file
    LLMF_DEBUG_PARAM_PATTERN=...  regex selecting which parameters to dump

Usage (already wired into `model_setup.prepare_training_components`):

    from debug_instrumentation import debug_enabled, install_debug_instrumentation
    if debug_enabled():
        install_debug_instrumentation(model, logger=logger)

Optional extras that require one line each in the trainer:

    # in train_fsdp.py / train_ddp.py, right after `create_optimizer(...)`:
    from debug_instrumentation import attach_optimizer_hooks
    attach_optimizer_hooks(optimizer)      # checks params before/after each step

    # in trainer.py, right before `optimizer_step(adam_lr, muon_lr, ...)`:
    from debug_instrumentation import check_gradients
    check_gradients(model, tag="pre-optimizer-step")
"""

import os
import re

import torch

# Module names we care about, matched on the class name so the same code works
# for `qwen3_5` (dense) and `qwen3_5_moe` without importing either.
_GDN_CLASS_MARKER = "GatedDeltaNet"
_GATED_NORM_CLASS_MARKER = "RMSNormGated"
_NORM_CLASS_MARKER = "RMSNorm"

# Transformers modules whose delta-rule / conv kernels we wrap so we can see the
# tensors immediately before and immediately after the Triton / CUDA kernel.
_KERNEL_HOST_MODULES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
)
_CHUNK_RULE_NAMES = ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule")
_RECURRENT_RULE_NAMES = ("torch_recurrent_gated_delta_rule", "recurrent_gated_delta_rule")
_CONV_NAMES = ("causal_conv1d_fn", "causal_conv1d_update")

_DEFAULT_PARAM_PATTERN = r"(A_log|dt_bias|in_proj_|out_proj|conv1d|norm|q_norm|k_norm)"

_SEPARATOR = "-" * 80
_BANNER = "=" * 80

# Singleton; `install_debug_instrumentation` sets it.
_INSTRUMENTATION = None


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def debug_enabled():
    """True when `LLMF_DEBUG` is set. Cheap enough to call unconditionally."""
    return _env_flag("LLMF_DEBUG", False)


def _global_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def _to_local(tensor):
    """Unwrap an FSDP2 `DTensor` to its local shard; pass anything else through."""
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        try:
            return to_local()
        except Exception:
            return tensor
    return tensor


class TensorStats:
    """Numerical summary of a single tensor. All fields are plain Python scalars."""

    __slots__ = (
        "shape",
        "dtype",
        "device",
        "numel",
        "min",
        "max",
        "mean",
        "std",
        "abs_max",
        "l2",
        "nan_pct",
        "inf_pct",
        "zero_pct",
        "finite_pct",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    @property
    def has_nan(self):
        return self.nan_pct > 0.0

    @property
    def has_inf(self):
        return self.inf_pct > 0.0

    @property
    def is_finite(self):
        return self.finite_pct >= 100.0

    def format(self, name, indent="  "):
        lines = [
            f"Tensor: {name}",
            f"{indent}shape={tuple(self.shape)}",
            f"{indent}dtype={self.dtype}",
            f"{indent}device={self.device}",
            f"{indent}min={self.min:.6g}",
            f"{indent}max={self.max:.6g}",
            f"{indent}mean={self.mean:.6g}",
            f"{indent}std={self.std:.6g}",
            f"{indent}abs_max={self.abs_max:.6g}",
            f"{indent}l2_norm={self.l2:.6g}",
            f"{indent}nan%={self.nan_pct:.4f}",
            f"{indent}inf%={self.inf_pct:.4f}",
            f"{indent}zero%={self.zero_pct:.4f}",
            f"{indent}finite%={self.finite_pct:.4f}",
        ]
        return "\n".join(lines)


def compute_tensor_stats(tensor):
    """
    Summarize `tensor` without perturbing it.

    min/max/mean/std/abs_max/l2 are computed over the *finite* entries only, so a
    tensor that is 1% NaN still reports the magnitude of the other 99% — which is
    what tells you whether the NaN came from an overflow or from a 0/0.
    """
    if not isinstance(tensor, torch.Tensor):
        return None

    local = _to_local(tensor).detach()
    if local.numel() == 0:
        return None

    work = local.reshape(-1)
    try:
        work = work.float()
    except RuntimeError:
        return None

    finite_mask = torch.isfinite(work)
    safe = torch.where(finite_mask, work, torch.zeros_like(work))
    n_finite = finite_mask.sum()
    denom = torch.clamp(n_finite, min=1).float()

    mean = safe.sum() / denom
    var = torch.clamp((safe * safe).sum() / denom - mean * mean, min=0.0)
    # `min`/`max` over finite entries: neutralize the masked-out slots.
    very_large = torch.finfo(torch.float32).max
    masked_min = torch.where(finite_mask, work, torch.full_like(work, very_large)).min()
    masked_max = torch.where(finite_mask, work, torch.full_like(work, -very_large)).max()

    # One device->host sync for every scalar instead of one per field.
    packed = torch.stack(
        [
            masked_min,
            masked_max,
            mean,
            torch.sqrt(var),
            safe.abs().max(),
            torch.linalg.vector_norm(safe, ord=2),
            torch.isnan(work).sum().float(),
            torch.isinf(work).sum().float(),
            (work == 0).sum().float(),
            n_finite.float(),
        ]
    ).tolist()

    numel = work.numel()
    scale = 100.0 / numel
    return TensorStats(
        shape=tuple(local.shape),
        dtype=local.dtype,
        device=local.device,
        numel=numel,
        min=packed[0],
        max=packed[1],
        mean=packed[2],
        std=packed[3],
        abs_max=packed[4],
        l2=packed[5],
        nan_pct=packed[6] * scale,
        inf_pct=packed[7] * scale,
        zero_pct=packed[8] * scale,
        finite_pct=packed[9] * scale,
    )


class DebugInstrumentation:
    """Owns the probe configuration, the anomaly latch, and all registered hooks."""

    def __init__(self, logger=None):
        self.logger = logger
        self.rank = _global_rank()
        self.all_ranks = _env_flag("LLMF_DEBUG_ALL_RANKS", False)
        self.active_rank = self.all_ranks or self.rank == 0

        self.verbose_steps = _env_int("LLMF_DEBUG_VERBOSE_STEPS", 2)
        self.dump_every = _env_int("LLMF_DEBUG_EVERY", 0)
        self.param_every = _env_int("LLMF_DEBUG_PARAM_EVERY", 50)
        self.explode_threshold = _env_float("LLMF_DEBUG_MAX_ABS", 1e4)
        self.vanish_threshold = _env_float("LLMF_DEBUG_SMALL_ABS", 1e-8)
        self.max_anomalies = _env_int("LLMF_DEBUG_MAX_ANOMALIES", 40)
        self.stop_on_nan = _env_flag("LLMF_DEBUG_STOP_ON_NAN", False)
        self.all_modules = _env_flag("LLMF_DEBUG_ALL_MODULES", False)
        self.param_pattern = re.compile(
            os.environ.get("LLMF_DEBUG_PARAM_PATTERN", _DEFAULT_PARAM_PATTERN)
        )

        log_path = os.environ.get("LLMF_DEBUG_FILE")
        self._file = open(log_path, "a", buffering=1) if (log_path and self.active_rank) else None  # noqa: SIM115

        self.iteration = 0
        self.event_id = 0
        self.anomalies_reported = 0
        self.first_nan = None
        self.first_inf = None
        self.first_nonfinite_param = None
        self.first_nonfinite_grad = None

        self._handles = []
        self._kernel_patches = []
        self._current_gdn = None  # (layer_idx, qualified_name) of the GDN in flight

    # Logging

    def _emit(self, text):
        if not self.active_rank:
            return
        prefix = f"[rank {self.rank}] " if self.all_ranks else ""
        message = "\n".join(prefix + line for line in text.splitlines())
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message, flush=True)
        if self._file is not None:
            self._file.write(message + "\n")

    def _header(self, site, phase):
        return f"[iter {self.iteration}] {site}  ({phase})"

    # Reporting

    def _should_dump(self):
        if self.iteration < self.verbose_steps:
            return True
        return self.dump_every > 0 and self.iteration % self.dump_every == 0

    def _classify(self, stats):
        flags = []
        if stats.has_nan:
            flags.append("NAN")
        if stats.has_inf:
            flags.append("INF")
        if stats.abs_max > self.explode_threshold:
            flags.append("EXPLODING")
        if stats.finite_pct > 0.0 and 0.0 < stats.abs_max < self.vanish_threshold:
            flags.append("VANISHING")
        return flags

    def report(self, site, phase, name, tensor, extra_lines=None):
        """Summarize `tensor` and print it if it is anomalous or we are in a dump window."""
        if not self.active_rank or not isinstance(tensor, torch.Tensor):
            return None

        stats = compute_tensor_stats(tensor)
        if stats is None:
            return None

        self.event_id += 1
        flags = self._classify(stats)

        if flags and (stats.has_nan or stats.has_inf):
            self._latch_first_nonfinite(site, phase, name, stats)

        if not flags and not self._should_dump():
            return stats

        if flags and self.anomalies_reported >= self.max_anomalies:
            return stats
        if flags:
            self.anomalies_reported += 1

        body = [_SEPARATOR, self._header(site, phase)]
        if flags:
            body.append(f"!! {' '.join(flags)} !!")
        body.append(stats.format(name))
        if extra_lines:
            body.extend(f"  {line}" for line in extra_lines)
        body.append(_SEPARATOR)
        self._emit("\n".join(body))

        if self.stop_on_nan and (stats.has_nan or stats.has_inf):
            raise RuntimeError(f"Non-finite tensor '{name}' in {site} during {phase}.")
        return stats

    def _latch_first_nonfinite(self, site, phase, name, stats):
        record = {
            "event": self.event_id,
            "iteration": self.iteration,
            "site": site,
            "phase": phase,
            "tensor": name,
        }
        newly_latched = []
        if stats.has_nan and self.first_nan is None:
            self.first_nan = record
            newly_latched.append("NaN")
        if stats.has_inf and self.first_inf is None:
            self.first_inf = record
            newly_latched.append("Inf")
        if not newly_latched:
            return

        self._emit(
            "\n".join(
                [
                    _BANNER,
                    f"FIRST {' / '.join(newly_latched)} DETECTED",
                    f"  iteration      : {self.iteration}",
                    f"  event ordinal  : {self.event_id}",
                    f"  phase          : {phase}",
                    f"  module         : {site}",
                    f"  tensor         : {name}",
                    f"  nan%           : {stats.nan_pct:.4f}",
                    f"  inf%           : {stats.inf_pct:.4f}",
                    f"  abs_max(finite): {stats.abs_max:.6g}",
                    f"  param already non-finite: {self.first_nonfinite_param or 'no'}",
                    f"  grad already non-finite : {self.first_nonfinite_grad or 'no'}",
                    _BANNER,
                ]
            )
        )

    # GDN-specific derived diagnostics

    def report_decay(self, site, g, chunk_size=64):
        """
        Cumulative-decay diagnostics for the gated delta rule.

        `g` is the per-token log-decay of shape (batch, seq, num_v_heads) and is
        always <= 0. The kernel takes `cumsum` within each chunk and exponentiates
        it, so what matters is the most negative *cumulative* value per chunk and
        how much of `exp(cumsum)` underflows to exactly zero.
        """
        if not self.active_rank or not isinstance(g, torch.Tensor) or g.ndim != 3:
            return
        if not (self.iteration < self.verbose_steps or self.dump_every > 0):
            return

        with torch.no_grad():
            work = _to_local(g).detach().float()
            seq_len = work.shape[1]
            pad = (chunk_size - seq_len % chunk_size) % chunk_size
            if pad:
                work = torch.nn.functional.pad(work, (0, 0, 0, pad))
            chunked = work.reshape(work.shape[0], -1, chunk_size, work.shape[-1])
            cumulative = chunked.cumsum(dim=2)
            decay = cumulative.exp()
            packed = torch.stack(
                [
                    cumulative.min(),
                    cumulative.max(),
                    (decay == 0).sum().float() * (100.0 / decay.numel()),
                    (decay > 0.999).sum().float() * (100.0 / decay.numel()),
                    torch.isfinite(cumulative).sum().float() * (100.0 / cumulative.numel()),
                ]
            ).tolist()

        self.report(
            site,
            "forward",
            "g (per-token log-decay)",
            g,
            extra_lines=[
                f"chunk_size={chunk_size}",
                f"cumsum_min={packed[0]:.6g}  (most negative in-chunk cumulative decay)",
                f"cumsum_max={packed[1]:.6g}",
                f"exp(cumsum)==0 %={packed[2]:.4f}  (decay underflow)",
                f"exp(cumsum)>0.999 %={packed[3]:.4f}  (no decay: state accumulates)",
                f"cumsum finite%={packed[4]:.4f}",
            ],
        )

    def report_beta(self, site, beta):
        """Delta-rule conditioning proxy: beta near 1 is the full-replacement regime."""
        if not self.active_rank or not isinstance(beta, torch.Tensor):
            return
        if not (self.iteration < self.verbose_steps or self.dump_every > 0):
            return
        with torch.no_grad():
            work = _to_local(beta).detach().float()
            total = 100.0 / max(work.numel(), 1)
            packed = torch.stack(
                [
                    (work > 0.99).sum().float() * total,
                    (work > 0.999).sum().float() * total,
                    (work < 0.01).sum().float() * total,
                ]
            ).tolist()
        self.report(
            site,
            "forward",
            "beta (delta-rule step size)",
            beta,
            extra_lines=[
                f"beta>0.99 %={packed[0]:.4f}  (ill-conditioned UT transform regime)",
                f"beta>0.999 %={packed[1]:.4f}",
                f"beta<0.01 %={packed[2]:.4f}",
            ],
        )

    # Parameter / gradient sweeps

    def dump_parameters(self, model, tag="parameters"):
        if not self.active_rank:
            return
        self._emit(f"{_BANNER}\n[iter {self.iteration}] {tag}\n{_BANNER}")
        for name, parameter in model.named_parameters():
            if not self.param_pattern.search(name):
                continue
            extra = None
            if name.endswith("A_log"):
                with torch.no_grad():
                    decay = _to_local(parameter).detach().float().exp()
                    bounds = torch.stack([decay.min(), decay.max()]).tolist()
                extra = [f"A=exp(A_log): min={bounds[0]:.6g} max={bounds[1]:.6g}"]
            self.report("parameters", "param", name, parameter, extra_lines=extra)

    def check_parameters(self, model, tag="parameters"):
        """Flag any parameter that has become non-finite. Returns the offending names."""
        offenders = []
        for name, parameter in model.named_parameters():
            local = _to_local(parameter).detach()
            if not torch.isfinite(local).all():
                offenders.append(name)
                if self.first_nonfinite_param is None:
                    self.first_nonfinite_param = f"{name} @ iter {self.iteration} ({tag})"
                self.report("parameters", f"param/{tag}", name, parameter)
        return offenders

    def check_gradients(self, model, tag="gradients"):
        """Flag any gradient that has become non-finite. Returns the offending names."""
        offenders = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            local = _to_local(parameter.grad).detach()
            if not torch.isfinite(local).all():
                offenders.append(name)
                if self.first_nonfinite_grad is None:
                    self.first_nonfinite_grad = f"{name} @ iter {self.iteration} ({tag})"
                self.report("gradients", f"grad/{tag}", f"{name}.grad", parameter.grad)
        return offenders

    def summary(self):
        self._emit(
            "\n".join(
                [
                    _BANNER,
                    "DEBUG INSTRUMENTATION SUMMARY",
                    f"  iterations observed      : {self.iteration}",
                    f"  tensors inspected        : {self.event_id}",
                    f"  anomalies reported       : {self.anomalies_reported}",
                    f"  first NaN                : {self.first_nan or 'none'}",
                    f"  first Inf                : {self.first_inf or 'none'}",
                    f"  first non-finite param   : {self.first_nonfinite_param or 'none'}",
                    f"  first non-finite grad    : {self.first_nonfinite_grad or 'none'}",
                    _BANNER,
                ]
            )
        )

    # Hook installation

    def _add(self, handle):
        self._handles.append(handle)

    def _probe_module(self, site, module, probe_backward=True):
        def forward_hook(_module, args, output):
            for index, item in enumerate(args):
                if isinstance(item, torch.Tensor):
                    self.report(site, "forward", f"input[{index}]", item)
            self._report_output(site, "forward", output)

        self._add(module.register_forward_hook(forward_hook, with_kwargs=False))

        if probe_backward:

            def backward_hook(_module, grad_input, grad_output):
                for index, item in enumerate(grad_output or ()):
                    if isinstance(item, torch.Tensor):
                        self.report(site, "backward", f"grad_output[{index}]", item)
                for index, item in enumerate(grad_input or ()):
                    if isinstance(item, torch.Tensor):
                        self.report(site, "backward", f"grad_input[{index}]", item)

            self._add(module.register_full_backward_hook(backward_hook))

    def _report_output(self, site, phase, output, prefix="output"):
        if isinstance(output, torch.Tensor):
            self.report(site, phase, prefix, output)
        elif isinstance(output, tuple | list):
            for index, item in enumerate(output):
                self._report_output(site, phase, item, prefix=f"{prefix}[{index}]")

    def _probe_gated_norm(self, site, module):
        """`RMSNormGated.forward(hidden_states, gate)` — capture both inputs and the output."""

        def pre_hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            gate = kwargs.get("gate", args[1] if len(args) > 1 else None)
            self.report(site, "forward", "rmsnorm_input", hidden)
            self.report(site, "forward", "rmsnorm_gate (z)", gate)

        def forward_hook(_module, args, kwargs, output):
            self.report(site, "forward", "rmsnorm_output", output)

        self._add(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._add(module.register_forward_hook(forward_hook, with_kwargs=True))
        self._probe_module(site, module, probe_backward=True)

    def _probe_gdn(self, site, module):
        layer_idx = getattr(module, "layer_idx", None)
        label = f"Layer {layer_idx} / {type(module).__name__}"

        def pre_hook(_module, args):
            self._current_gdn = label

        def post_hook(_module, args, output):
            self._report_output(label, "forward", output, prefix="gdn_output")
            self._current_gdn = None

        self._add(module.register_forward_pre_hook(pre_hook))
        self._add(module.register_forward_hook(post_hook))
        self._probe_module(label, module, probe_backward=True)

        for child_name, child in module.named_children():
            child_site = f"{label}.{child_name}"
            if _GATED_NORM_CLASS_MARKER in type(child).__name__:
                self._probe_gated_norm(child_site, child)
            else:
                self._probe_module(child_site, child, probe_backward=True)

    def _install_param_grad_hooks(self, model):
        """
        Fire the moment a parameter's `.grad` is accumulated — i.e. during backward
        and therefore strictly before `optimizer.step()`.
        """

        def make_hook(name):
            def hook(parameter):
                if parameter.grad is None:
                    return
                local = _to_local(parameter.grad).detach()
                if torch.isfinite(local).all():
                    return
                if self.first_nonfinite_grad is None:
                    self.first_nonfinite_grad = f"{name} @ iter {self.iteration} (post-accumulate)"
                self.report("gradients", "backward/post-accumulate", f"{name}.grad", parameter.grad)

            return hook

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            register = getattr(parameter, "register_post_accumulate_grad_hook", None)
            if register is None:
                continue
            self._add(register(make_hook(name)))

    def _install_kernel_patches(self):
        """
        Wrap the delta-rule and causal-conv kernels so we see every tensor
        immediately before it enters the Triton/CUDA kernel and immediately after
        it comes back out. Backward hooks on those same tensors tell us whether the
        corruption is produced inside the kernel or in the surrounding PyTorch code.
        """
        import importlib

        for module_path in _KERNEL_HOST_MODULES:
            try:
                host = importlib.import_module(module_path)
            except (ImportError, ModuleNotFoundError):
                continue

            for name in _CHUNK_RULE_NAMES + _RECURRENT_RULE_NAMES:
                original = getattr(host, name, None)
                if callable(original):
                    setattr(host, name, self._wrap_delta_rule(name, original))
                    self._kernel_patches.append((host, name, original))

            for name in _CONV_NAMES:
                original = getattr(host, name, None)
                if callable(original):
                    setattr(host, name, self._wrap_conv(name, original))
                    self._kernel_patches.append((host, name, original))

    def _tap_backward(self, site, name, tensor):
        """Report the gradient flowing back through `tensor`."""
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return
        tensor.register_hook(lambda grad: self.report(site, "backward", f"d/d {name}", grad))

    def _wrap_delta_rule(self, kernel_name, original):
        def wrapped(*args, **kwargs):
            site = f"{self._current_gdn or 'GatedDeltaNet'} :: {kernel_name}"
            query = kwargs.get("query", args[0] if len(args) > 0 else None)
            key = kwargs.get("key", args[1] if len(args) > 1 else None)
            value = kwargs.get("value", args[2] if len(args) > 2 else None)
            g = kwargs.get("g", args[3] if len(args) > 3 else None)
            beta = kwargs.get("beta", args[4] if len(args) > 4 else None)
            initial_state = kwargs.get("initial_state")

            self.report(site, "pre-kernel", "query (post-conv)", query)
            self.report(site, "pre-kernel", "key (post-conv)", key)
            self.report(site, "pre-kernel", "value (post-conv)", value)
            self.report_beta(site, beta)
            self.report_decay(site, g, chunk_size=int(kwargs.get("chunk_size", 64) or 64))
            self.report(site, "pre-kernel", "initial_state", initial_state)

            for name, tensor in (
                ("query", query),
                ("key", key),
                ("value", value),
                ("g", g),
                ("beta", beta),
            ):
                self._tap_backward(site, name, tensor)

            output = original(*args, **kwargs)

            core_attn_out, last_recurrent_state = None, None
            if isinstance(output, tuple | list) and len(output) >= 2:
                core_attn_out, last_recurrent_state = output[0], output[1]
            else:
                core_attn_out = output

            self.report(site, "post-kernel", "core_attn_out", core_attn_out)
            self.report(site, "post-kernel", "recurrent_state", last_recurrent_state)
            self._tap_backward(site, "core_attn_out", core_attn_out)
            return output

        return wrapped

    def _wrap_conv(self, kernel_name, original):
        def wrapped(*args, **kwargs):
            site = f"{self._current_gdn or 'GatedDeltaNet'} :: {kernel_name}"
            hidden = kwargs.get("hidden_states", kwargs.get("x", args[0] if args else None))
            self.report(site, "pre-kernel", "conv_input (mixed_qkv)", hidden)
            self._tap_backward(site, "conv_input", hidden)
            output = original(*args, **kwargs)
            self.report(site, "post-kernel", "conv_output", output)
            self._tap_backward(site, "conv_output", output)
            return output

        return wrapped

    def install(self, model):
        gdn_count = 0
        for name, module in model.named_modules():
            class_name = type(module).__name__
            if _GDN_CLASS_MARKER in class_name:
                self._probe_gdn(name or class_name, module)
                gdn_count += 1
            elif _GATED_NORM_CLASS_MARKER in class_name:
                continue  # already covered as a GDN child
            elif (
                _NORM_CLASS_MARKER in class_name
                and self.all_modules
                or self.all_modules
                and isinstance(module, torch.nn.Linear | torch.nn.Embedding)
            ):
                self._probe_module(name or class_name, module, probe_backward=False)

        self._install_param_grad_hooks(model)
        self._install_kernel_patches()

        # Root pre-forward hook drives the iteration counter and periodic dumps.
        def root_pre_hook(_module, _args):
            self.iteration += 1
            if self.param_every > 0 and self.iteration % self.param_every == 0:
                self.dump_parameters(model, tag=f"parameter sweep @ iter {self.iteration}")
                self.check_parameters(model, tag="periodic")

        self._add(model.register_forward_pre_hook(root_pre_hook))

        self._emit(
            "\n".join(
                [
                    _BANNER,
                    "DEBUG INSTRUMENTATION INSTALLED",
                    f"  GatedDeltaNet modules probed : {gdn_count}",
                    f"  kernels wrapped              : {len(self._kernel_patches)}",
                    f"  verbose iterations           : {self.verbose_steps}",
                    f"  dump every                   : {self.dump_every or 'off'}",
                    f"  parameter sweep every        : {self.param_every or 'off'}",
                    f"  explode / vanish thresholds  : {self.explode_threshold:g} / "
                    f"{self.vanish_threshold:g}",
                    f"  active ranks                 : {'all' if self.all_ranks else 'rank 0'}",
                    _BANNER,
                ]
            )
        )
        self.dump_parameters(model, tag="parameter sweep @ initialization")
        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for host, name, original in self._kernel_patches:
            setattr(host, name, original)
        self._kernel_patches.clear()
        if self._file is not None:
            self._file.close()
            self._file = None


def install_debug_instrumentation(model, logger=None):
    """Install the probes on `model`. Call once, before FSDP/DDP wrapping."""
    global _INSTRUMENTATION
    if _INSTRUMENTATION is not None:
        return _INSTRUMENTATION
    _INSTRUMENTATION = DebugInstrumentation(logger=logger).install(model)
    return _INSTRUMENTATION


def get_instrumentation():
    return _INSTRUMENTATION


def check_gradients(model, tag="manual"):
    """Optional explicit sweep, e.g. right before `optimizer_step(...)`."""
    if _INSTRUMENTATION is None:
        return []
    return _INSTRUMENTATION.check_gradients(model, tag=tag)


def check_parameters(model, tag="manual"):
    if _INSTRUMENTATION is None:
        return []
    return _INSTRUMENTATION.check_parameters(model, tag=tag)


def attach_optimizer_hooks(optimizer):
    """
    Bracket `optimizer.step()` so we can tell whether parameters were already
    corrupt on entry (backward is to blame) or only on exit (the update is).
    """
    if _INSTRUMENTATION is None:
        return optimizer

    state = _INSTRUMENTATION

    def pre_hook(opt, args, kwargs):
        for group in opt.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not torch.isfinite(_to_local(parameter.grad).detach()).all():
                    if state.first_nonfinite_grad is None:
                        state.first_nonfinite_grad = f"<pre-step> @ iter {state.iteration}"
                    state.report("optimizer", "pre-step", "grad", parameter.grad)
                    return

    def post_hook(opt, args, kwargs):
        for group in opt.param_groups:
            for parameter in group["params"]:
                if not torch.isfinite(_to_local(parameter).detach()).all():
                    if state.first_nonfinite_param is None:
                        state.first_nonfinite_param = f"<post-step> @ iter {state.iteration}"
                    state.report("optimizer", "post-step", "param", parameter)
                    return

    optimizer.register_step_pre_hook(pre_hook)
    optimizer.register_step_post_hook(post_hook)
    return optimizer


def summary():
    if _INSTRUMENTATION is not None:
        _INSTRUMENTATION.summary()
