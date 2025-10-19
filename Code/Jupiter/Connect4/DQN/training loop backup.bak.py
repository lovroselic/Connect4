# training loop backup.bak
# === Training loop  ===

start_time = time.time()
env_steps = 0

with tqdm(total=num_episodes, desc="Training Episodes", leave=True, dynamic_ncols=True) as pbar:
    for episode in range(num_episodes):
        
        state = env.reset()
        nstep_buf.reset()     # important: clear rolling window per episode
        ply_idx = 0
        total_reward = 0      # terminal reward only
        final_result = None   # 1 win, -1 loss, 0.5 draw
        agent.center_forced_used = False
        first_logged = False
        
        # --- handle phase & hyperparams ---   
        new_phase, strategy_weights, epsilon, memory_prune_low, epsilon_min, target_update_interval, \
                TU_mode, TU_tau, guard_prob, center_start, seed_fraction, lr = get_phase(episode)
        
        phase_changed = new_phase != phase

        if phase_changed and phase is not None:
            agent.update_target_model() 
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_path = f"{MODEL_DIR}{TRAINING_SESSION}_Temp_Connect4 dqn_model_{timestamp} episodes-{episode+1} phase-{phase}.pt"
            default_model_path = f"phase-{phase} Interim Connect4 DQN model.pt"
            
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.model.state_dict(), default_model_path)
            print(f"Temp Model saved to {model_path}")
        
        phase, frozen_opp = handle_phase_change(agent, new_phase, phase, epsilon, memory_prune_low, epsilon_min, frozen_opp,
            env, episode, device, Lookahead, TRAINING_SESSION, guard_prob, center_start, seed_fraction, lr)

        prev_mode = getattr(agent, "target_update_mode", None)
        prev_tu   = getattr(agent, "target_update_interval", None)
        
        if phase_changed or (TU_mode != prev_mode) or (target_update_interval != prev_tu):
            agent.reset_target_update_timer(env_steps)
            agent.guard_prob = guard_prob
            agent.center_start = center_start 
            agent.target_update_mode = TU_mode
            agent.tau = TU_tau
            agent.set_target_update_interval(target_update_interval)
            agent.update_target_model() 
            
        if phase_changed and openings is not None:
            openings.mark_phase(episode + 1, new_phase)

        # --------- main training loop ---------
        # alternating start
        if episode % 2 == 1:
            env.current_player = -1
            env_steps +=1
            agent.maybe_update_target(env_steps) 
            s0 = state
            opp_action = get_opponent_action(env, agent, episode, s0, depth=lookahead_depth, frozen_opp=frozen_opp, phase=phase)
            s1, r_opp, _ = env.step(opp_action)
            # append the opponent step
            nstep_buf.append(s0, opp_action, r_opp, s1, env.done, player=-1)
            state = s1

            if ply_idx == 0:
                record_first_move(opening_kpis, opp_action, who="opp")
                openings.on_first_move(int(opp_action), is_agent=False)
            ply_idx += 1

        if env.done:
            final_result = evaluate_final_result(env, agent_player=+1)
            total_reward = map_final_result_to_reward(final_result)

        # Turn-by-turn until terminal
        while not env.done:
            # --- Agent (+1) acts ---
            env.current_player = +1
            env_steps +=1
            agent.maybe_update_target(env_steps) 
            valid_actions = env.available_actions()

            if len(valid_actions) == 0: raise ValueError(f"No valid actions in agents turn, done {env.done}, step: {env.ply}")
            action = agent.act(state, valid_actions,depth=lookahead_depth, strategy_weights=strategy_weights, ply = ply_idx)
            
            next_state, r_agent, _ = env.step(action)

            # append agent step
            nstep_buf.append(state, action, r_agent, next_state, env.done, player=+1)

            if ply_idx == 0:  # agent started this game
                record_first_move(opening_kpis, action, who="agent")
                openings.on_first_move(int(action), is_agent=True)    
            ply_idx += 1

            if env.done:
                # terminal on agent move
                final_result = evaluate_final_result(env, agent_player=+1)
                total_reward = map_final_result_to_reward(final_result)
                state = next_state
                break

            # --- Opponent (−1) responds ---
            env.current_player = -1
            env_steps +=1
            agent.maybe_update_target(env_steps) 
            opp_action = get_opponent_action(env, agent, episode, next_state, depth=lookahead_depth, frozen_opp=frozen_opp, phase=phase)
            next_state2, r_opp, _ = env.step(opp_action)

            # append opponent step
            nstep_buf.append(next_state, opp_action, r_opp, next_state2, env.done, player=-1)

            if env.done:
                final_result = evaluate_final_result(env, agent_player=+1)
                total_reward = map_final_result_to_reward(final_result)
                state = next_state2
                break

            # continue
            state = next_state2

        # Episode ended — flush remaining short tails
        nstep_buf.flush()

        # --- Training updates ---

        mem_len = len(agent.memory)
        MIX = getattr(agent, "per_mix_1step", 0.70)
        
        # Warmup thresholds (enough diversity before full-sized batches)
        WARMUP_HALF = max(batch_size * 2, 256)
        WARMUP_FULL = max(batch_size * 5, 1024)
        
        if mem_len >= WARMUP_FULL:
            # Full batches; number of updates scales with episode length (ply)
            updates = 4 + min(12, env.ply // 2)           # 4..16 updates per episode
            for _ in range(updates):
                agent.replay(batch_size, mix_1step=MIX)
        
        elif mem_len >= WARMUP_HALF:
            # Smaller batches + fewer updates during early buffer growth
            b = max(32, min(batch_size, mem_len // 16))   # adaptive mini-batch
            updates = 2 + min(6, env.ply // 3)            # 2..8 updates
            for _ in range(updates):
                agent.replay(b, mix_1step=MIX)
        
        else:
            # Tiny trickle updates to get losses/priors moving without overfitting seeds
            if mem_len >= 64:
                for _ in range(2):
                    agent.replay(32, mix_1step=MIX)



        # --------- history bookkeeping ---------
        epsilon_history.append(agent.epsilon)
        epsilon_min_history.append(agent.epsilon_min)
        reward_history.append(total_reward)
        memory_prune_low_history.append(memory_prune_low)
        guard_prob_history.append(agent.guard_prob)
        center_prob_history.append(agent.center_start)
        tu_interval_history.append(agent.target_update_interval)
        tau_history.append(agent.tau)
        tu_mode_history.append(agent.target_update_mode)
        openings.maybe_log(episode + 1)

        # Epsilon decay (per episode)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        # Win/loss/draw counters
        wins, losses, draws = track_result(final_result, win_history)
        win_count += wins
        loss_count += losses
        draw_count += draws

        # periodic benchmarking
        if (episode + 1) % BENCHMARK_EVERY == 0 or episode == 0:
            benchmark_history = update_benchmark_winrates(
                agent, env, device, Lookahead, episode, benchmark_history,
                save=f"{LOG_DIR}DQN-{TRAINING_SESSION}-benchmark.xlsx"  
            )

        #### Live plot ###
        # rewards and co
        if (episode + 1) % plot_interval == 0:
            avg_reward = np.mean(reward_history[-25:]) if len(reward_history) >= 1 else 0.0
            pbar.set_postfix(avg_reward=f"{avg_reward:.2f}", epsilon=f"{agent.epsilon:.3f}",
                             wins=win_count, losses=loss_count, draws=draw_count, phase=phase)


            fig_ep, fig_step = plot_live_training(
                episode, reward_history, win_history, epsilon_history,
                phase, win_count, loss_count, draw_count, TRAINING_SESSION,
                epsilon_min_history, memory_prune_low_history, agent=agent,
                bench_history=benchmark_history, bench_smooth_k=3,
                tu_interval_history=tu_interval_history,
                tau_history=tau_history,
                tu_mode_history=tu_mode_history,
                guard_prob_history=guard_prob_history, center_prob_history=center_prob_history,
                openings=openings,
                return_figs=True
            )

        if fig_ep is not None:
            plots_handle.update(fig_ep)   
            plt.close(fig_ep)
            
        if fig_step is not None:
            steps_handle.update(fig_step)
            plt.close(fig_step)

        # Periodic logging
        if (episode + 1) % log_every_x_episode == 0:
            log_summary_stats(
                episode=episode, reward_history=reward_history, win_history=win_history, phase=phase,
                strategy_weights=strategy_weights, agent=agent, win_count=win_count, loss_count=loss_count,
                draw_count=draw_count, summary_stats_dict=summary_stats
            )
            open_summary = summarize_opening_kpis(opening_kpis)
            summary_stats["openings"] = {
                "summary": open_summary,
                "episodes": openings.episodes,
                "agent_center_rate": openings.a_center_rates,
                "opp_center_rate": openings.o_center_rates,
            }
            
        pbar.update(1)

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTraining completed in {elapsed/60:.1f} minutes ({elapsed / num_episodes:.2f} s/episode)")


