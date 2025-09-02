# dqn_utilities.py

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from DQN.training_phases_config import TRAINING_PHASES
from C4.connect4_env import Connect4Env
from DQN.eval_utilities import log_phase_evaluation

evaluation_opponents = {
    "Random": 203,
    "Lookahead-1": 101,
    "Lookahead-2": 53,
    "Lookahead-3": 11,
}

def get_phase(episode):
    for phase_name, phase_data in TRAINING_PHASES.items():
        end_episode = phase_data["length"]
        if end_episode is None or episode < end_episode:
            return (
                phase_name,
                phase_data["weights"],
                phase_data["epsilon"],
                phase_data["memory_prune_low"],
                phase_data.get("epsilon_min", 0.0),
            )


def handle_phase_change(agent, new_phase, current_phase, epsilon, memory_prune_low, epsilon_min, prev_frozen_opp, env, episode, device, Lookahead, training_session):
    frozen_opp = prev_frozen_opp 
    
    if new_phase.startswith("SelfPlay"): 
        if frozen_opp is None:
            frozen_opp = deepcopy(agent)
            frozen_opp.model.eval()
            frozen_opp.target_model.eval()
            frozen_opp.epsilon = 0.0
            frozen_opp.epsilon_min = 0.0
    else:
        frozen_opp = None
    
    if new_phase != current_phase:
        agent.epsilon = epsilon
        agent.epsilon_min = epsilon_min
        agent.memory.prune(memory_prune_low, mode="low_priority")
        
        log_phase_evaluation(
            agent, env, phase_name=current_phase, episode=episode,
            device=device, Lookahead=Lookahead,
            evaluation_opponents=evaluation_opponents, excel_path= f"{training_session}-training_session.xlsx"
            )
        
        return new_phase, frozen_opp

    return current_phase, frozen_opp


def evaluate_final_result(env, agent_player):
    if env.winner == agent_player:
        return 1
    elif env.winner == -agent_player:
        return -1
    elif env.winner == 0:
        return 0.5
    else:
        return None


def map_final_result_to_reward(final_result):
    if final_result == 1:
        return Connect4Env.WIN_REWARD
    elif final_result == -1:
        return Connect4Env.LOSS_PENALTY
    elif final_result == 0.5:
        return Connect4Env.DRAW_REWARD
    else:
        return 0


def track_result(final_result, win_history):
    if final_result == 1:
        win_history.append(1)
        return 1, 0, 0
    elif final_result == -1:
        win_history.append(0)
        return 0, 1, 0
    elif final_result == 0.5:
        win_history.append(0)
        return 0, 0, 1
    else:
        print("DEBUG:", final_result, type(final_result))
        print(f"Final result before track_result: {final_result}")

        raise ValueError("Invalid final_result — env.winner was not set correctly. Lovro, get a grip!")


def plot_moving_averages(ax, data, windows, colors=None, styles=None, labels=None):
    for i, win in enumerate(windows):
        if len(data) >= win:
            avg = np.convolve(data, np.ones(win) / win, mode='valid')
            offset = win - 1
            ax.plot(range(offset, len(data)), avg,
                    label=labels[i] if labels else f'{win}-ep Avg',
                    color=colors[i] if colors else None,
                    linestyle=styles[i] if styles else None)


def plot_live_training(
    episode, reward_history, win_history, epsilon_history, phase,
    win_count, loss_count, draw_count, title,
    epsilon_min_history, memory_prune_low_history,
    agent=None, save=False, path=None
):
    # === Episode-based metrics ===
    fig_ep, ax_ep = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # --- Reward ---
    plot_moving_averages(
        ax_ep[0], reward_history, windows=[25, 100, 500],
        colors=['blue', '#666', '#000'],
        styles=['-', '--', '--'],
        labels=['25-ep', '100-ep', '500-ep']
    )
    ax_ep[0].plot(reward_history, label='Reward', alpha=0.4)
    ax_ep[0].set_ylabel('Reward')
    ax_ep[0].legend(loc='lower left', fontsize=8)
    ax_ep[0].grid(True)

    # --- Win Rate + Epsilon Overlay ---
    plot_moving_averages(
        ax_ep[1], win_history, windows=[25, 100, 250, 500],
        colors=['green', 'red', '#999', '#11F'],
        styles=['-', '-', '--', (0, (1, 5))],
        labels=['25-ep', '100-ep', '250-ep', '500-ep']
    )

    # Overlay epsilon on same axis (scaled 0–1, same as win rate)
    ax_ep[1].plot(epsilon_history, label='Epsilon', color='orange', linestyle='--', alpha=0.7)
    
    # Labeling
    ax_ep[1].set_ylabel('Win Rate')
    ax_ep[1].legend(loc='lower left', fontsize=8)
    ax_ep[1].grid(True)

    # --- Epsilon ---
    ax_ep[2].plot(epsilon_history, label='Epsilon', color='orange')
    ax_ep[2].plot(epsilon_min_history, label='Epsilon_min', color='black', linestyle='--')
    ax_ep[2].set_ylabel('Epsilon')
    ax_ep[2].legend(loc='lower left', fontsize=8)
    ax_ep[2].grid(True)

    # --- Memory Prune ---
    ax_ep[3].plot(memory_prune_low_history, label='Memory Prune', color='red')
    ax_ep[3].set_ylabel('Mem Prune')
    ax_ep[3].legend(loc='lower left', fontsize=8)
    ax_ep[3].grid(True)

    # --- Phase transition markers (bottom-left labels) ---
    ep_pos = 0.01  # X offset
    y_factor = 0.05  # fraction of y-range from bottom

    for name, meta in TRAINING_PHASES.items():
        transition_ep = meta.get("length", None)
        if transition_ep is not None and transition_ep <= episode:
            for axis in ax_ep:
                axis.axvline(transition_ep, color='black', linestyle='dotted', linewidth=1)

                ymin, ymax = axis.get_ylim()
                y_pos = ymin + (ymax - ymin) * y_factor
                axis.text(transition_ep + ep_pos, y_pos, name, rotation=90,
                          va='bottom', ha='left', fontsize=8, color='black')

    # --- Title ---
    fig_ep.suptitle(
        f"{title} - Ep {episode} — Phase: {phase} | Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count} | ε={epsilon_history[-1]:.3f}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    display(fig_ep)

    # === Step-based metrics (unshared x-axis) ===
    if agent:
        plots = []
        if hasattr(agent, "loss_hist") and agent.loss_hist:
            plots.append(("Loss", agent.loss_hist, "purple"))
        if hasattr(agent, "td_hist") and agent.td_hist:
            plots.append(("TD Error", agent.td_hist, "gray"))
        if hasattr(agent, "per_w_hist") and agent.per_w_hist:
            plots.append(("IS Weights", agent.per_w_hist, "brown"))
    
        if plots:
            fig_step, ax_step = plt.subplots(len(plots), 1, figsize=(12, 3 * len(plots)))
            if len(plots) == 1:
                ax_step = [ax_step]  # ensure iterable
    
            for i, (label, data, color) in enumerate(plots):
                ax = ax_step[i]
                ax.plot(data, label=label, color=color, alpha=0.6)
    
                # Add moving average only for Loss
                if label == "Loss" and len(data) >= 100:
                    ma = np.convolve(data, np.ones(100) / 100, mode='valid')
                    ax.plot(range(99, len(data)), ma, label='100-ep MA', color='black', linestyle='--')
    
                ax.set_ylabel(label)
                ax.legend()
                ax.grid(True)
    
            fig_step.suptitle("Step-based Metrics")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            display(fig_step)

            if save and path is not None:
                fig_ep.savefig(f"{path}{title}_episode_metrics.png")
                fig_step.savefig(f"{path}{title}_step_metrics.png")
                print(f"Plots saved to {path}")

    plt.close('all')




# === FINAL SUMMARY PLOT ===
def save_final_winrate_plot(win_history, training_phases, save_path, session_name="default_session"):
    if len(win_history) < 25:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    win_avg = np.convolve(win_history, np.ones(25)/25, mode='valid')
    ax.plot(range(24, len(win_history)), win_avg, label='Win Rate (25 ep)', color='green')

    ax.set_title("Final Win Rate Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.grid(True)
    ax.legend()

    for name, meta in training_phases.items():
        transition_ep = meta["length"]
        if transition_ep is not None and transition_ep <= len(win_history):
            ax.axvline(transition_ep, color='black', linestyle='dotted', linewidth=1)
            ax.text(transition_ep + 2, ax.get_ylim()[1] * 0.95, name, rotation=90, va='top', ha='left', fontsize=8)

    full_path = f"{save_path}DQN-{session_name}_final_winrate.png"
    fig.savefig(full_path)
    plt.close(fig)


# === PHASE-WISE SUMMARY LOGGING ===
def log_summary_stats(
    episode,
    reward_history,
    win_history,
    phase,
    strategy_weights,
    agent,
    win_count,
    loss_count,
    draw_count,
    summary_stats_dict
):
    recent_rewards = reward_history[-25:] if len(reward_history) >= 25 else reward_history
    recent_win_rate = np.mean(win_history[-25:]) if len(win_history) >= 25 else np.mean(win_history)

    summary_stats_dict[episode] = {
        "phase": phase,
        "strategy_weights": strategy_weights.copy() if strategy_weights else None,
        "epsilon": agent.epsilon,
        "wins": win_count,
        "losses": loss_count,
        "draws": draw_count,
        "avg_reward_25": round(np.mean(recent_rewards), 2),
        "win_rate_25": round(recent_win_rate, 2)
    }
