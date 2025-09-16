# A0/a0_runner.py
# End-to-end training runner with:
# - Live plotting (per-step losses/diagnostics + per-cycle eval points)
# - Excel logging
# - Checkpointing (save + resume from named checkpoint, including replay buffer)
# - Short interim evals + optional final benchmark & bar chart

from __future__ import annotations
import os, time, torch
from typing import Optional
from tqdm.auto import tqdm

from A0.az_env import EnvAdapter
from A0.a0_utilities import state_encoder
from A0.az_net import AZNet
from A0.a0_mcts import MCTS
from A0.a0_buffer import ReplayBuffer
from A0.a0_train import AZTrainer
from A0.a0_loop import selfplay_cycle, train_cycle
from A0.a0_eval import eval_vs_random, eval_vs_l1
from A0.a0_plot import LiveDiagnostics
from A0.a0_ckpt import save_checkpoint, restore_from_checkpoint
from A0.a0_logger_xlsx import ExcelLogger
from A0.a0_bench import evaluate_suite
import A0.a0_config as C


def run_training(
    out_dir: str = "./A0_runs/run1",
    max_cycles: int = 50,
    eval_every: int = 2,
    ckpt_every: int = 2,
    final_eval: bool = True,
    final_suite: Optional[list[str]] = None,
    # --- resume options ---
    resume_ckpt: Optional[str] = None,
    resume_load_buffer: bool = True,
    resume_strict: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Components
    env    = EnvAdapter(state_encoder=state_encoder)
    model  = AZNet().to(device).eval()
    mcts   = MCTS(model, env, sims=C.MCTS_SIMS, c_puct=C.CPUCT, device=device)
    buffer = ReplayBuffer(capacity=C.REPLAY_CAPACITY)
    trainer= AZTrainer(model, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
                       grad_clip=C.GRAD_CLIP, device=device, use_amp=C.USE_AMP)

    # ----- Resume from checkpoint (optional) -----
    base_cycle = 0
    global_updates = 0
    resume_meta = {}
    if resume_ckpt is not None:
        meta, rb_state = restore_from_checkpoint(
            resume_ckpt,
            model,
            optimizer=trainer.optimizer,
            scaler=getattr(trainer, "scaler", None),
            device=device,
            strict=resume_strict,
        )
        resume_meta = meta or {}
        base_cycle = int(resume_meta.get("cycle", 0))
        global_updates = int(resume_meta.get("global_updates", 0))
        if resume_load_buffer and rb_state is not None:
            buffer.load_state_dict(rb_state)
        print(f"[resume] loaded '{resume_ckpt}' | meta={resume_meta} | buffer_len={len(buffer)}")

    # Diagnostics
    live = LiveDiagnostics()
    train_fields = ["time","cycle","buffer","games","avg_moves","tr_updates",
                    "tr_loss","tr_policy","tr_value","ent_model","ent_target","top1_match"]
    eval_fields  = ["time","cycle","wr_random","wr_l1","W_R","L_R","D_R","W_L1","L_L1","D_L1"]
    xlsx_path = os.path.join(out_dir, "logs.xlsx")
    train_xl = ExcelLogger(xlsx_path, "train", train_fields)
    eval_xl  = ExcelLogger(xlsx_path, "eval",  eval_fields)

    t0 = time.time()

    # ----- Outer progress across cycles -----
    for cyc in tqdm(range(1, max_cycles + 1), desc="Cycles", disable=not C.USE_TQDM):
        gcyc = base_cycle + cyc  # global cycle index (continues when resuming)

        # --- Self-play (short per-cycle tqdm lives inside) ---
        def on_game(games_done, s):
            live.update_selfplay(gcyc, games_done, s["wins"], s["losses"], s["draws"], s["avg_moves"])
            live.refresh()

        sp = selfplay_cycle(
            env, mcts, buffer,
            games=C.GAMES_PER_CYCLE,
            t_switch=C.TAU_SWITCH_PLY, tau=C.TAU_EARLY, tau_final=C.TAU_LATE,
            augment=C.AUGMENT, use_tqdm=C.USE_TQDM,
            on_game=on_game, desc_suffix=f"(cycle {gcyc}/{base_cycle+max_cycles})"
        )

        # --- Train (live curves per step) ---
        live.start_cycle(gcyc)

        def on_step(step_idx, stats):
            live.update_step(gcyc, step_idx, stats)
            live.refresh()

        tr = train_cycle(
            trainer, buffer,
            updates=C.UPDATES_PER_CYCLE,
            batch_size=C.BATCH_SIZE,
            device=device, use_tqdm=C.USE_TQDM,
            on_step=on_step, desc_suffix=f"(cycle {gcyc}/{base_cycle+max_cycles})"
        )
        global_updates += int(tr.get("tr_updates") or 0)

        # --- Log + per-cycle points ---
        train_xl.log({
            "time": float(f"{time.time()-t0:.1f}"),
            "cycle": gcyc,
            "buffer": len(buffer),
            "games": sp["sp_games"],
            "avg_moves": float(sp["sp_avg_moves"]),
            "tr_updates": tr["tr_updates"],
            "tr_loss": tr["tr_loss"],
            "tr_policy": tr["tr_policy"],
            "tr_value": tr["tr_value"],
            "ent_model": tr["ent_model"],
            "ent_target": tr["ent_target"],
            "top1_match": tr["top1_match"],
        }, train_fields)
        live.update_train(gcyc, tr)
        live.refresh()

        # --- Interim eval (short; plotted as dots) ---
        if cyc % eval_every == 0:
            n_games = C.EVAL_GAMES_INTERIM
            er  = eval_vs_random(env, mcts, games=n_games, eval_sims=C.EVAL_MCTS_SIMS)
            el1 = eval_vs_l1(env, mcts, games=n_games, eval_sims=C.EVAL_MCTS_SIMS)
            eval_xl.log({
                "time": float(f"{time.time()-t0:.1f}"), "cycle": gcyc,
                "wr_random": er["win_rate"], "wr_l1": el1["win_rate"],
                "W_R": er["W"], "L_R": er["L"], "D_R": er["D"],
                "W_L1": el1["W"], "L_L1": el1["L"], "D_L1": el1["D"]
            }, eval_fields)
            live.update_eval_point(gcyc, er["win_rate"], el1["win_rate"])
            live.refresh()

        # --- Checkpoint ---
        if cyc % ckpt_every == 0:
            save_checkpoint(
                model, trainer.optimizer, getattr(trainer, "scaler", None),
                gcyc, global_updates, len(buffer),
                os.path.join(out_dir, f"ckpt_cycle_{gcyc:04d}.pt"),
                replay_buffer_state=buffer.state_dict(),   # include replay buffer for seamless resume
                extra={"config": {k: getattr(C, k) for k in dir(C) if k.isupper()}},
            )

    train_xl.close(); eval_xl.close()

    # ------------- FINAL BENCHMARK (optional; short/long set in config) -------------
    final_results = None
    if final_eval:
        suite = ["random", "L1", "L2"] if final_suite is None else final_suite
        final_results = evaluate_suite(
            env, mcts,
            opponent_ids=suite,
            games_each=C.EVAL_GAMES_FINAL,
            eval_sims=C.EVAL_MCTS_SIMS,
            out_xlsx=xlsx_path, sheet="final_eval", seed=123
        )
        # Display bar chart
        live.show_final_eval_bars(final_results)

    live.refresh()
    return {
        "out_dir": out_dir,
        "xlsx": xlsx_path,
        "resume_meta": resume_meta,
        "start_cycle": base_cycle + 1,
        "end_cycle": base_cycle + max_cycles,
        "final": final_results,
    }
