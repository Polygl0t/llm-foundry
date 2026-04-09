"""
Optimizer factories and learning-rate schedulers for the DDP trainer.

Provides:
    - Muon Stuff:
        - get_muon_momentum:             compute the Muon momentum for a given step
        - zeropower_via_newtonschulz5:   compute the Muon orthogonalization via NS iteration
        - muon_update:                   compute the Muon update
        - Muon:                          the Muon optimizer class for distributed training
        - SingleDeviceMuon:              the Muon optimizer class for single-device training
        - MuonWithAuxAdam:               hybrid optimizer for distributed training with Muon and Adam
        - SingleDeviceMuonWithAuxAdam:   hybrid optimizer for single-device training with Muon and Adam

    - Supporting functions for optimizer construction and learning-rate scheduling:
        - normalize_optimizer_type:      validate / canonicalize an optimizer name
        - create_lr_scheduler:           build a cosine or WSD learning-rate schedule
        - create_optimizer:              build AdamW or MuonWithAuxAdam with its step fn
        - get_optimizer_summary_lines:   formatted config lines for logging
"""
import torch
import math

import numpy as np

import torch
import torch.distributed as dist

def get_muon_momentum(it):
    """
    Compute the Muon momentum for a given step. The momentum:
        - starts at 0.85 and; 
        - increases to 0.95 over the first 300 iterations.
        - It follows a cosine schedule.
    """
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    https://kellerjordan.github.io/posts/muon/
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    
OPTIMIZER_ALIASES = {
    "adam": "adamw",
    "adamw": "adamw",
    "muon": "muon_adam",
    "muon_adam": "muon_adam",
    "muon+adam": "muon_adam",
    "hybrid": "muon_adam",
}

def normalize_optimizer_type(optimizer_type):
    normalized = (optimizer_type or "adamw").strip().lower()
    normalized = OPTIMIZER_ALIASES.get(normalized, normalized)
    if normalized not in {"adamw", "muon_adam"}:
        supported = ", ".join(sorted({"adamw", "muon_adam"}))
        raise ValueError(f"Invalid optimizer type: '{optimizer_type}'. Supported types are: {supported}.")
    return normalized

def create_lr_scheduler(args, max_steps, optimizer_type):
    def cosine_schedule(it, max_lr):
        lr_decay_iters = max_steps * args.lr_decay_iters_coef
        if args.warmup_steps > 0 and it < args.warmup_steps:
            return max_lr * (it + 1) / args.warmup_steps, "warmup"
        if it > lr_decay_iters:
            return args.min_learning_rate, "stable"

        decay_ratio = (it - args.warmup_steps) / (lr_decay_iters - args.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_learning_rate + coeff * (max_lr - args.min_learning_rate), "cosine_decay"

    def wsd_schedule(it, max_lr):
        lr_decay_iters = max_steps * args.lr_decay_iters_coef
        stable_iters = max_steps - lr_decay_iters
        if args.warmup_steps > 0 and it < args.warmup_steps:
            return max_lr * (it + 1) / args.warmup_steps, "warmup"
        if it > stable_iters and lr_decay_iters > 0:
            decay_ratio = (it - stable_iters) / (max_steps - stable_iters)
            assert 0 <= decay_ratio <= 1
            if args.use_sqrt:
                decay_ratio = np.sqrt(decay_ratio)
            coeff = 1.0 - decay_ratio
            stage = "linear_decay" if not args.use_sqrt else "1-sqrt"
            return args.min_learning_rate + coeff * (max_lr - args.min_learning_rate), stage
        return max_lr, "stable"

    if args.lr_decay_type.lower() == "cosine":
        schedule_fn = cosine_schedule
    elif args.lr_decay_type.lower() == "wsd":
        schedule_fn = wsd_schedule
    else:
        raise ValueError(f"Invalid learning rate decay type: '{args.lr_decay_type}'. Supported types are: `cosine` and `wsd`.")

    def lr_scheduler(it):
        adam_lr, stage = schedule_fn(it, args.max_learning_rate)
        muon_lr = None
        if optimizer_type == "muon_adam":
            muon_lr, _ = schedule_fn(it, args.muon_learning_rate)
        return adam_lr, muon_lr, stage

    return lr_scheduler

def create_optimizer(model, args, device_type, optimizer_type, master_process, logger=None):
    no_decay = ["bias", "layer_norm.weight", "embed_tokens.weight"]

    if optimizer_type == "muon_adam":
        hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed_tokens.weight" not in n]
        embed_params = [p for n, p in model.named_parameters() if "embed_tokens.weight" in n]
        scalar_params_with_decay = [p for n, p in model.named_parameters() if p.ndim < 2 and not any(nd in n for nd in no_decay)]
        scalar_params_no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 and any(nd in n for nd in no_decay)]

        optimizer_grouped_parameters = [
            {
                "params": embed_params,
                "weight_decay": 0.0,
                "lr": args.max_learning_rate,
                "betas": (args.beta1, args.beta2),
                "eps": args.eps,
                "use_muon": False,
            },
            {
                "params": scalar_params_no_decay,
                "weight_decay": 0.0,
                "lr": args.max_learning_rate,
                "betas": (args.beta1, args.beta2),
                "eps": args.eps,
                "use_muon": False,
            },
            {
                "params": scalar_params_with_decay,
                "weight_decay": args.weight_decay,
                "lr": args.max_learning_rate,
                "betas": (args.beta1, args.beta2),
                "eps": args.eps,
                "use_muon": False,
            },
            {
                "params": hidden_matrix_params,
                "weight_decay": args.weight_decay,
                "lr": args.muon_learning_rate,
                "momentum": args.beta2,
                "use_muon": True,
            }
        ]
        optimizer = MuonWithAuxAdam(optimizer_grouped_parameters)
        optimizer_label = "Adam + Muon"

        if args.torch_compile:
            if master_process and logger is not None:
                logger.info("Compiling optimizer step with torch.compile.")

            @torch.compile(fullgraph=False)
            def optimizer_step(adam_lr, muon_lr, step):
                for param_group in optimizer.param_groups:
                    if param_group["use_muon"]:
                        param_group["lr"] = muon_lr
                        param_group["momentum"] = get_muon_momentum(step)
                    else:
                        param_group["lr"] = adam_lr
                optimizer.step()
        else:
            def optimizer_step(adam_lr, muon_lr, step):
                for param_group in optimizer.param_groups:
                    if param_group["use_muon"]:
                        param_group["lr"] = muon_lr
                        param_group["momentum"] = get_muon_momentum(step)
                    else:
                        param_group["lr"] = adam_lr
                optimizer.step()
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.max_learning_rate,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
            fused=True if device_type == "cuda" else False,
        )
        optimizer_label = "AdamW"

        if args.torch_compile:
            if master_process and logger is not None:
                logger.info("Compiling optimizer step with torch.compile.")

            @torch.compile(fullgraph=False)
            def optimizer_step(adam_lr, _muon_lr, _step):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adam_lr
                optimizer.step()
        else:
            def optimizer_step(adam_lr, _muon_lr, _step):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adam_lr
                optimizer.step()

    return optimizer, optimizer_step, optimizer_label

def get_optimizer_summary_lines(args, optimizer_type):
    summary_lines = [
        f"  Optimizer type | {optimizer_type}",
        f"  Max learning rate (Adam) | {args.max_learning_rate}",
        f"  Min learning rate | {args.min_learning_rate}",
        f"  LR scheduler type | {args.lr_decay_type.upper()}",
        f"  LR decay iterations coef | {args.lr_decay_iters_coef}",
        f"  Warmup steps | {args.warmup_steps}",
        f"  Weight decay | {args.weight_decay}",
        f"  Beta1 | {args.beta1}",
        f"  Beta2 | {args.beta2}",
        f"  Epsilon | {args.eps}",
        f"  Max grad norm | {args.max_grad_norm}",
    ]
    if optimizer_type == "muon_adam":
        summary_lines.insert(2, f"  Muon learning rate | {args.muon_learning_rate}")
    return summary_lines
