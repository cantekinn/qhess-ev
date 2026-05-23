"""Ortak training / evaluation runner.

train_one_episode: bir episode RL eğitimi
evaluate_policy:   greedy aksiyon ile bir episode test, history döndür
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from src.env.hess_env import HESSEnv


def train_one_episode(env: HESSEnv, agent, max_steps: int | None = None, start_idx: int | None = None) -> dict:
    s = env.reset(start_idx=start_idx)
    total_r = 0.0
    n_viol = 0
    loss_sum = 0.0
    step = 0
    max_steps = max_steps or env.cfg.max_steps
    while True:
        a = agent.act(s, greedy=False)
        s_next, r, done, info = env.step(a)
        l = agent.update(s, a, r, s_next, done)
        try:
            loss_sum += float(l) if l is not None else 0.0
        except Exception:
            pass
        total_r += r
        n_viol += int(info.get("violation", False))
        s = s_next
        step += 1
        if done or step >= max_steps:
            break
    if hasattr(agent, "end_episode"):
        agent.end_episode()
    return {
        "total_reward": total_r,
        "violations": n_viol,
        "steps": step,
        "mean_loss": loss_sum / max(1, step),
    }


def evaluate_policy(env: HESSEnv, policy: Callable[[np.ndarray], int] | object, start_idx: int = 0,
                    max_steps: int | None = None, soc_bat_init: float = 0.85, soc_sc_init: float = 0.70) -> list[dict]:
    """Greedy/sabit politikayı bir epizot boyunca koştur, env.history döndür."""
    if hasattr(policy, "reset"):
        policy.reset()
    s = env.reset(start_idx=start_idx, soc_bat_init=soc_bat_init, soc_sc_init=soc_sc_init)
    max_steps = max_steps or env.cfg.max_steps
    step = 0
    while True:
        if hasattr(policy, "act"):
            a = policy.act(s, greedy=True)
        else:
            a = int(policy(s))
        s_next, r, done, info = env.step(int(a))
        s = s_next if s_next is not None else s
        step += 1
        if done or step >= max_steps:
            break
    return list(env.history)
