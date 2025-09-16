# A0/a0_loop.py
from __future__ import annotations
from typing import Dict, Callable, Optional
#import numpy as np
import torch
from tqdm import tqdm

from A0.a0_selfplay import play_game
from A0.a0_buffer import ReplayBuffer

def selfplay_cycle(
    env, mcts, buffer: ReplayBuffer,
    games: int,
    t_switch: int, tau: float, tau_final: float,
    augment: bool,
    use_tqdm: bool = True, on_game=None, desc_suffix: str = ""
) -> Dict[str, float]:
    wins = losses = draws = 0
    total_moves = 0

    it = tqdm(range(games), desc=f"Self-play {desc_suffix}".strip(), leave=False, disable=not use_tqdm)
    
    for i in it:
        steps, winner = play_game(
            env, mcts, to_play=+1,
            t_switch=t_switch, tau=tau, tau_final=tau_final, max_moves=42
        )
        buffer.add_game(steps, augment=augment)
        total_moves += len(steps)

        if winner == +1: wins += 1
        elif winner == -1: losses += 1
        else: draws += 1

        done = i + 1
        avg_moves = total_moves / max(1, done)
        it.set_postfix(W=wins, L=losses, D=draws, avg_moves=f"{avg_moves:.1f}")

        if on_game is not None:
            on_game(done, {"wins": wins, "losses": losses, "draws": draws, "avg_moves": avg_moves})

    return {
        "sp_games": games,
        "sp_avg_moves": total_moves / max(1, games),
        "sp_win": wins, "sp_loss": losses, "sp_draw": draws
    }

def train_cycle(
    trainer,
    buffer: ReplayBuffer,
    updates: int,
    batch_size: int,
    device: torch.device,
    use_tqdm: bool = True,
    on_step: Optional[Callable[[int, Dict[str, float]], None]] = None,
    desc_suffix: str = "",
) -> Dict[str, float]:
    if len(buffer) == 0:
        return {"tr_updates": 0, "tr_loss": None, "tr_policy": None, "tr_value": None,
                "ent_model": None, "ent_target": None, "top1_match": None}

    tot = pol = val = ent_m = ent_t = top1 = 0.0
    it = tqdm(range(updates), desc=f"Train {desc_suffix}".strip(), leave=False, disable=not use_tqdm)
    steps_done = 0
    for i in it:
        batch = buffer.sample_batch(batch_size=batch_size, device=device, replace=True)
        stats = trainer.train_step(batch)
        steps_done += 1

        tot  += stats["loss_total"]; pol  += stats["loss_policy"]; val  += stats["loss_value"]
        ent_m+= stats.get("ent_model", 0.0); ent_t+= stats.get("ent_target", 0.0); top1 += stats.get("top1_match", 0.0)

        it.set_postfix(loss=f"{stats['loss_total']:.3f}",
                       pol=f"{stats['loss_policy']:.3f}",
                       val=f"{stats['loss_value']:.3f}",
                       acc=f"{stats.get('top1_match',0.0):.2f}",
                       Hm=f"{stats.get('ent_model',0.0):.2f}")
        if on_step: on_step(i + 1, stats)

    return {
        "tr_updates": steps_done,
        "tr_loss":   tot / steps_done,
        "tr_policy": pol / steps_done,
        "tr_value":  val / steps_done,
        "ent_model": ent_m / steps_done,
        "ent_target":ent_t / steps_done,
        "top1_match":top1 / steps_done,
    }
