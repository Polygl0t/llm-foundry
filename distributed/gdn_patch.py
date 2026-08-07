"""
GatedDeltaNet (Qwen3.5) initialization patch for the distributed trainers.

Fixes a latent initialization bug in the upstream `transformers`
`Qwen3_5GatedDeltaNet` / `Qwen3_5MoeGatedDeltaNet` layers. The decay
parameter `A_log` is created as:

    A = torch.empty(num_v_heads).uniform_(0, 16)   # inherits the model dtype
    A_log = nn.Parameter(torch.log(A))

Because the tensor inherits the model's (bf16) default dtype rather than the
float32 used by the reference Mamba-2 / Qwen3-Next code, `uniform_(0, 16)` has
coarse resolution near zero and rounds a draw down to exactly `0.0` with
probability ~0.4% per head in bf16, so `torch.log(0) = -inf`. The affected head
then has `A = exp(-inf) = 0` (a decay gate that never forgets) and its
gradient `dg/dA_log = -A * softplus(...) = 0`, so it stays frozen at `-inf`
for the whole run.

The probability of hitting this scales with `num_v_heads * num_gdn_layers`,
which is why a, e.g., a 35B hybrid (32 value heads x 30 GDN layers -> ~4 dead
heads expected) trips it while a tiny config (8 x 8) does not.

This module re-initializes / repairs `A_log` (and `dt_bias`) so they are
always finite. It is applied to the freshly built random model before it is
sharded, so it never touches genuinely trained checkpoints.

Usage (wired into `model_setup.py`):

    from gdn_patch import patch_qwen3_5_gdn_initialization

    # from-scratch build (master rank, before the bootstrap checkpoint is saved):
    patch_qwen3_5_gdn_initialization(random_model, force_reinit=True)

    # defensive, every rank, before FSDP wrapping (no-op on healthy models):
    patch_qwen3_5_gdn_initialization(model, logger=logger, force_reinit=False)
"""

import hashlib

import torch

# Matched on the class name so the same code works for `qwen3_5` (dense) and
# `qwen3_5_moe` without importing either modeling module.
_GDN_CLASS_MARKER = "GatedDeltaNet"

# Reference (Mamba-2 / Qwen3-Next) sampling range for the decay strength A.
_A_UNIFORM_LOW = 0.0
_A_UNIFORM_HIGH = 16.0
# Floor applied before log() so a near-zero draw can never yield A_log = -inf.
_A_MIN = 1e-4


def _module_generator(name: str, base_seed: int) -> torch.Generator:
    """Deterministic per-module RNG so re-initialization is identical on every
    rank, regardless of where the patch is invoked."""
    digest = hashlib.blake2b(f"{base_seed}:{name}".encode(), digest_size=8).digest()
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(digest, "big") % (2**63 - 1))
    return generator


def _mark_no_weight_decay(module: torch.nn.Module) -> None:
    """Mirror the upstream convention so decay-free optimizers can also key off
    the attribute, not just the parameter name."""
    for attr in ("A_log", "dt_bias"):
        param = getattr(module, attr, None)
        if isinstance(param, torch.nn.Parameter):
            param._no_weight_decay = True


def _reinit_decay_params(module, name, base_seed, cast_to_fp32) -> int:
    """Draw a fresh, guaranteed-finite `A_log` (and reset `dt_bias` to ones)."""
    num_v_heads = module.A_log.numel()
    generator = _module_generator(name, base_seed)

    a = torch.empty(num_v_heads, dtype=torch.float32).uniform_(
        _A_UNIFORM_LOW, _A_UNIFORM_HIGH, generator=generator
    )
    a.clamp_(min=_A_MIN)

    target_dtype = torch.float32 if cast_to_fp32 else module.A_log.dtype
    with torch.no_grad():
        new_a_log = torch.log(a).to(device=module.A_log.device, dtype=target_dtype)
        module.A_log = torch.nn.Parameter(new_a_log, requires_grad=module.A_log.requires_grad)

        dt_bias = getattr(module, "dt_bias", None)
        if isinstance(dt_bias, torch.nn.Parameter):
            dt_dtype = torch.float32 if cast_to_fp32 else dt_bias.dtype
            new_dt = torch.ones(dt_bias.numel(), device=dt_bias.device, dtype=dt_dtype)
            module.dt_bias = torch.nn.Parameter(new_dt, requires_grad=dt_bias.requires_grad)

    _mark_no_weight_decay(module)
    return num_v_heads


def _repair_decay_params(module) -> int:
    """Replace only the non-finite entries of `A_log` / `dt_bias` in place.

    Deterministic (no RNG), so it is safe to run on every rank after loading a
    checkpoint that identical weights across ranks."""
    repaired = 0
    with torch.no_grad():
        a_log = module.A_log
        finite = torch.isfinite(a_log)
        if not bool(finite.all()):
            fallback = a_log[finite].min() if bool(finite.any()) else a_log.new_zeros(())
            a_log[~finite] = fallback
            repaired += int((~finite).sum().item())

        dt_bias = getattr(module, "dt_bias", None)
        if isinstance(dt_bias, torch.nn.Parameter):
            bad = ~torch.isfinite(dt_bias)
            if bool(bad.any()):
                dt_bias[bad] = 1.0
                repaired += int(bad.sum().item())

    if repaired:
        _mark_no_weight_decay(module)
    return repaired


def _log(logger, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def patch_qwen3_5_gdn_initialization(
    model,
    *,
    logger=None,
    force_reinit: bool = False,
    cast_decay_params_to_fp32: bool = False,
    base_seed: int = 1234,
) -> int:
    """Fix the GatedDeltaNet decay-parameter initialization on a Qwen3.5 model.

    Args:
        model: any module tree; GatedDeltaNet layers are located by class name.
        logger: optional `logging.Logger`; falls back to `print`.
        force_reinit: `True` re-draws `A_log` and resets `dt_bias` (use on a
            freshly built random model). `False` only repairs non-finite entries
            in place (safe, deterministic, no-op on healthy / trained checkpoints).
        cast_decay_params_to_fp32: keep `A_log` / `dt_bias` as float32 leaves
            instead of the model dtype (only meaningful when `force_reinit`).
        base_seed: seed feeding the deterministic per-layer RNG.

    Returns:
        The number of head entries re-initialized (force) or repaired.
    """
    gdn_modules = [
        (name, module)
        for name, module in model.named_modules()
        if _GDN_CLASS_MARKER in type(module).__name__ and hasattr(module, "A_log")
    ]
    if not gdn_modules:
        return 0

    touched_layers = 0
    total = 0
    for name, module in gdn_modules:
        if force_reinit:
            total += _reinit_decay_params(module, name, base_seed, cast_decay_params_to_fp32)
            touched_layers += 1
        else:
            repaired = _repair_decay_params(module)
            if repaired:
                total += repaired
                touched_layers += 1

    if force_reinit:
        _log(
            logger,
            f"GatedDeltaNet decay init patch: re-initialized A_log/dt_bias on "
            f"{len(gdn_modules)} layer(s) "
            f"({'fp32' if cast_decay_params_to_fp32 else 'model dtype'} leaves).",
        )
    elif total:
        _log(
            logger,
            f"GatedDeltaNet decay init patch: repaired {total} non-finite decay "
            f"entr{'y' if total == 1 else 'ies'} across {touched_layers} of "
            f"{len(gdn_modules)} layer(s).",
        )
    return total
