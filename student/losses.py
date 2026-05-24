"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    """
    GRU-friendly one-step delta loss.

    For MLP, this is still valid.
    For GRU, this preserves temporal hidden-state evolution instead of flattening all steps.
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
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    per_step = (pred_norm - target_norm).pow(2).mean(dim=(0, 2))
    H = per_step.shape[0]
    weights = torch.linspace(1.0, 0.8, H, device=per_step.device)
    return F.mse_loss(pred_norm, target_norm)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]

    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(model, states, actions, normalizer)

    warmup = int(cfg["eval"].get("warmup_steps", 10))

    main_horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=main_horizon,
    )

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.0))

    long_w = float(loss_cfg.get("long_rollout_weight", 0.0))
    long_h = int(loss_cfg.get("long_rollout_horizon", main_horizon))

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

    total = one_w * one + roll_w * roll + long_w * long_roll

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/long_rollout": float(long_roll.detach().cpu()),
    }