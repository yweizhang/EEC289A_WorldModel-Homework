"""Student one-step plus rollout loss with continuous cosine curriculum learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math

from .rollout import open_loop_rollout

# Stateful tracker to monitor total updates across epochs without resetting
_GLOBAL_UPDATE_STEP = 0

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    """
    GRU-friendly one-step delta loss.
    Preserves temporal hidden-state evolution instead of flattening all steps.
    """
    B, T_plus_1, obs_dim = states.shape
    T = actions.shape[1]

    hidden = model.initial_hidden(B, states.device)
    losses = []

    for t in range(T):
        obs = states[:, t]
        act = actions[:, t]
        target_delta = states[:, t + 1] - states[:, t]

        obs_norm = normalizer.normalize_obs(obs)
        act_norm = normalizer.normalize_act(act)
        target_norm = normalizer.normalize_delta(target_delta)

        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        losses.append(F.mse_loss(pred_norm, target_norm))

    return torch.stack(losses).mean()


def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        horizon = max(1, states.shape[1] - int(warmup_steps) - 1)
        needed_states = int(warmup_steps) + int(horizon) + 1

    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0

    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    pred_states = open_loop_rollout(
        model,
        sub_states,
        sub_actions,
        normalizer,
        warmup_steps=warmup_steps,
        horizon=horizon,
    )

    target_states = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(pred_states)
    target_norm = normalizer.normalize_obs(target_states)

    per_step = (pred_norm - target_norm).pow(2).mean(dim=(0, 2))

    H = per_step.shape[0]
    steps = torch.linspace(0.0, 1.0, H, device=per_step.device)
    weights = 1.0 + 1.0 * (1.0 - torch.cos(torch.pi * steps)) / 2.0

    return (per_step * weights).sum() / weights.sum()

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    global _GLOBAL_UPDATE_STEP
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]

    states = batch["states"]
    actions = batch["actions"]

    # Calculate 1-step delta baseline error
    one = one_step_delta_loss(model, states, actions, normalizer)
    warmup = int(cfg["eval"].get("warmup_steps", 10))

    # -----------------------------------------------------------------
    # FIX 3: SMOOTH PROGRESSIVE CURRICULUM FORMULATION
    # -----------------------------------------------------------------
    total_expected_updates = float(train_cfg.get("updates", 8000))
    progress = min(float(_GLOBAL_UPDATE_STEP) / (0.65 * total_expected_updates), 1.0)
    
    # Half-cosine curve transitions smoothly from 0.0 to 1.0
    cos_factor = 0.5 * (1.0 - math.cos(math.pi * progress))

    # Curve A: Horizon scaling configuration
    base_horizon = 10
    target_horizon = int(loss_cfg.get("rollout_train_horizon", 25))
    current_horizon = int(base_horizon + cos_factor * (target_horizon - base_horizon))
    current_horizon = max(1, min(current_horizon, target_horizon))

    # Curve B: Dynamic Loss Weighting curves
    base_rollout_w = 0.3
    target_rollout_w = float(loss_cfg.get("rollout_weight", 0.7))
    current_rollout_weight = base_rollout_w + cos_factor * (target_rollout_w - base_rollout_w)

    # -----------------------------------------------------------------
    # Execution using scheduled targets
    # -----------------------------------------------------------------
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=current_horizon,
    )

    one_w = float(loss_cfg.get("one_step_weight", 1.8))
    long_w = float(loss_cfg.get("long_rollout_weight", 0.0))
    long_h = int(loss_cfg.get("long_rollout_horizon", current_horizon))

    if long_w > 0.0:
        long_roll = rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup,
            horizon=long_h,
        )
    else:
        long_roll = torch.zeros((), device=states.device)

    # Combine regular terms + scale with dynamic weights
    total = (one_w * one) + (current_rollout_weight * roll) + (long_w * long_roll)

    # Increment update step safety counter
    _GLOBAL_UPDATE_STEP += 1

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/long_rollout": float(long_roll.detach().cpu()),
        "curriculum/horizon": float(current_horizon),
        "curriculum/rollout_weight": float(current_rollout_weight)
    }
