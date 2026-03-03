from __future__ import annotations


def evaluate_greedy(env, agent, n_episodes: int = 200, stage: str = "C") -> dict[str, float]:
    capture_count = 0
    steps_to_capture = 0
    total_reward = 0.0
    invalid_moves = 0
    total_steps = 0

    for _ in range(n_episodes):
        state = env.reset(stage=stage)
        done = False
        reward_sum = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, training=False)
            nxt, reward, done, info = env.step(action)
            state = nxt
            reward_sum += reward
            steps += 1
            total_steps += 1
            if info["invalid_move"]:
                invalid_moves += 1

            if done and info["event"] == "capture":
                capture_count += 1
                steps_to_capture += steps

        total_reward += reward_sum

    capture_rate = capture_count / n_episodes
    avg_steps_capture = steps_to_capture / max(capture_count, 1)
    invalid_rate = invalid_moves / max(total_steps, 1)

    return {
        "capture_rate": capture_rate,
        "avg_steps_to_capture": avg_steps_capture,
        "avg_reward": total_reward / n_episodes,
        "invalid_move_rate": invalid_rate,
    }
