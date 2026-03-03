from __future__ import annotations

from typing import Callable

from config import ASSETS_DIR
from metrics import save_training_curves


def build_stage_schedule(curriculum: list[tuple[str, int]], episodes: int) -> list[str]:
    schedule: list[str] = []
    for stage, count in curriculum:
        schedule.extend([stage] * count)
    if len(schedule) < episodes:
        schedule.extend([curriculum[-1][0]] * (episodes - len(schedule)))
    return schedule[:episodes]

# Train the agent across staged difficulty and record learning history.
def train(env, agent, episodes: int, curriculum: list[tuple[str, int]], callback: Callable | None = None) -> dict[str, list[float]]:
    schedule = build_stage_schedule(curriculum, episodes)

    history: dict[str, list[float]] = {
        "episode_rewards": [],
        "episode_steps": [],
        "capture_flags": [],
        "epsilon_values": [],
        "avg_distance": [],
    }

    for epi in range(1, episodes + 1):
        stage = schedule[epi - 1]
        state = env.reset(stage=stage)
        done = False

        total_reward = 0.0
        step_count = 0
        capture = 0.0
        dist_sum = 0.0

        while not done:
            action = agent.select_action(state, training=True)
            nxt, reward, done, info = env.step(action)
            agent.update_q(state, action, reward, nxt, done)
            state = nxt

            step_count += 1
            total_reward += reward
            dist_sum += float(info["distance"])

            if done and info["event"] == "capture":
                capture = 1.0

            if callback is not None:
                callback(
                    {
                        "mode": "train",
                        "episode": epi,
                        "stage": stage,
                        "step": step_count,
                        "total_reward": total_reward,
                        "epsilon": agent.epsilon,
                        "event": info["event"],
                        "capture": bool(capture),
                        "distance": info["distance"],
                        "line_of_sight": info["line_of_sight"],
                        "render_data": env.render_data(),
                        "action": action,
                    }
                )

        agent.decay_epsilon()

        history["episode_rewards"].append(total_reward)
        history["episode_steps"].append(float(step_count))
        history["capture_flags"].append(capture)
        history["epsilon_values"].append(agent.epsilon)
        history["avg_distance"].append(dist_sum / max(step_count, 1))

    save_training_curves(history, ASSETS_DIR)
    return history
