# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:50:36 2025

@author: LOVRO
"""

# dqn_utilities.py

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from DQN.training_phases_config import TRAINING_PHASES


def get_phase(episode):
    for phase_name, phase_data in TRAINING_PHASES.items():
        end_episode = phase_data["length"]
        if end_episode is None or episode < end_episode:
            return (
                phase_name,
                phase_data["weights"],
                phase_data["epsilon"],
                phase_data["memory_prune"]
            )

def handle_phase_change(agent, new_phase, current_phase, epsilon, memory_prune):
    if new_phase != current_phase:
        agent.epsilon = epsilon
        if memory_prune:
            agent.memory.prune(memory_prune)
        return new_phase
    return current_phase

def evaluate_final_result(env, agent_player):
    if env.winner == agent_player:
        return 1
    elif env.winner == -agent_player:
        return -1
    elif env.winner == 0:
        return 0.5
    else:
        return None

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
        raise ValueError("Invalid final_result — env.winner was not set correctly.")

def plot_live_training(episode, reward_history, win_history, epsilon_history, phase, win_count, loss_count, draw_count, title="title", memory_prune = 0):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Plot reward
    ax[0].plot(reward_history, label='Reward')
    if len(reward_history) >= 25:
        avg = np.convolve(reward_history, np.ones(25) / 25, mode='valid')
        ax[0].plot(range(24, len(reward_history)), avg, label='25-ep Moving Avg', linewidth=2)
    if len(reward_history) >= 100:
        avg = np.convolve(reward_history, np.ones(100) / 100, mode='valid')
        ax[0].plot(range(99, len(reward_history)), avg, label='100-ep Moving Avg', linewidth=1, linestyle='--')
    if len(reward_history) >= 500:
        avg = np.convolve(reward_history, np.ones(500) / 500, mode='valid')
        ax[0].plot(range(499, len(reward_history)), avg, label='500-ep Moving Avg', linewidth=1, color='#000', linestyle='--')
    ax[0].set_ylabel('Total Reward')
    ax[0].legend()
    ax[0].grid(True)

    # Win rate
    if len(win_history) >= 25:
        win_avg = np.convolve(win_history, np.ones(25) / 25, mode='valid')
        ax[1].plot(range(24, len(win_history)), win_avg, label='Win Rate (25 ep)', color='green')
    if len(win_history) >= 250:
        win_avg = np.convolve(win_history, np.ones(250) / 250, mode='valid')
        ax[1].plot(range(249, len(win_history)), win_avg, label='Win Rate (250 ep)', color='#999', linestyle='--')
    ax[1].set_ylabel('Win Rate')
    ax[1].legend()
    ax[1].grid(True)

    # Epsilon
    ax[2].plot(epsilon_history, label='Epsilon', color='orange')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Epsilon')
    ax[2].legend()
    ax[2].grid(True)

    fig.suptitle(f"{title} - Episode {episode} — Phase: {phase} | Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count} | ε={epsilon_history[-1]:.3f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for name, meta in TRAINING_PHASES.items():
        transition_ep = meta["length"]
        if transition_ep is not None and transition_ep <= episode:
            for axis in ax:
                axis.axvline(transition_ep, color='black', linestyle='dotted', linewidth=1)
                axis.text(transition_ep + 2, axis.get_ylim()[1] * 0.95, name, rotation=90, va='top', ha='left', fontsize=8)

    display(fig)
    plt.close()
    

def save_final_winrate_plot(
    win_history,
    training_phases,
    save_path,
    session_name="default_session"
):
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
            ax.text(
                transition_ep + 2,
                ax.get_ylim()[1] * 0.95,
                name,
                rotation=90,
                va='top',
                ha='left',
                fontsize=8
            )

    full_path = f"{save_path}DQN-{session_name}_final_winrate.png"
    fig.savefig(full_path)
    plt.close(fig)
    
    
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
        "strategy_weights": strategy_weights.copy(),
        "epsilon": agent.epsilon,
        "wins": win_count,
        "losses": loss_count,
        "draws": draw_count,
        "avg_reward_25": round(np.mean(recent_rewards), 2),
        "win_rate_25": round(recent_win_rate, 2)
    }


    



