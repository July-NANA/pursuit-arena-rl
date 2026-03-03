from __future__ import annotations

import argparse
import json

from agent import QLearningAgent
from config import (
    ALPHA,
    ASSETS_DIR,
    BASELINE_MODEL_PATH,
    CURRICULUM,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    EVAL_EPISODES,
    GAMMA,
    MODEL_PATH,
    N_ACTIONS,
    N_STATES,
    TARGET_CAPTURE_RATE,
    TRAIN_EPISODES,
)
from env import PursuitArenaEnv
from evaluate import evaluate_greedy
from metrics import save_comparison_bar, save_training_curves
from train import train


def build_agent(algorithm: str = "double") -> QLearningAgent:
    return QLearningAgent(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        algorithm=algorithm,
    )


def sanity_check_env() -> dict[str, bool]:
    env = PursuitArenaEnv(seed=7)
    checks: dict[str, bool] = {}

    env.reset(stage="B")
    env.enemy_pos = (0, 0)
    env.player_pos = (9, 9)
    _, reward, done, info = env.step(0)
    checks["invalid_move_has_penalty"] = (reward < 0.0) and (not done) and info["invalid_move"]

    env.reset(stage="B")
    env.enemy_pos = (5, 5)
    env.player_pos = (5, 6)
    _, reward, done, info = env.step(3)
    checks["capture_done_plus20"] = done and info["event"] == "capture" and abs(reward - 20.0) < 1e-6

    env.reset(stage="B")
    env.steps = env.max_steps - 1
    env.enemy_pos = (0, 0)
    env.player_pos = (9, 9)
    _, _, done, info = env.step(1)
    checks["timeout_done"] = done and info["event"] == "timeout"

    return checks


def run_train(save: bool, algorithm: str, episodes: int) -> None:
    env = PursuitArenaEnv(seed=42, reward_mode="full")
    agent = build_agent(algorithm=algorithm)

    checks = sanity_check_env()
    print("Sanity checks:")
    print(json.dumps(checks, indent=2))

    history = train(env, agent, episodes=episodes, curriculum=CURRICULUM)
    save_training_curves(history, ASSETS_DIR, prefix=algorithm)

    if save:
        if algorithm == "double":
            agent.save(str(MODEL_PATH))
        else:
            agent.save(str(BASELINE_MODEL_PATH))

    print(f"Training finished ({algorithm}). episodes={episodes} epsilon={agent.epsilon:.4f}")


def run_eval(load: bool, algorithm: str, stage: str, n_episodes: int) -> None:
    env = PursuitArenaEnv(seed=42, reward_mode="full")
    agent = build_agent(algorithm=algorithm)

    if load:
        model_path = MODEL_PATH if algorithm == "double" else BASELINE_MODEL_PATH
        if model_path.exists():
            agent.load(str(model_path))

    metrics = evaluate_greedy(env, agent, n_episodes=n_episodes, stage=stage)
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Target capture rate >= {TARGET_CAPTURE_RATE:.2f}: {metrics['capture_rate'] >= TARGET_CAPTURE_RATE}")


def run_ui(ui_mode: str, load: bool, algorithm: str) -> None:
    from ui import run_demo

    env = PursuitArenaEnv(seed=42, reward_mode="full")
    agent = build_agent(algorithm=algorithm)
    if load:
        model_path = MODEL_PATH if algorithm == "double" else BASELINE_MODEL_PATH
        if model_path.exists():
            agent.load(str(model_path))
    run_demo(env, agent, mode=ui_mode)


def run_compare(episodes: int, eval_episodes: int) -> None:
    results: dict[str, dict[str, float]] = {}

    env_base = PursuitArenaEnv(seed=42, reward_mode="full")
    agent_base = build_agent("baseline")
    hist_base = train(env_base, agent_base, episodes=episodes, curriculum=CURRICULUM)
    save_training_curves(hist_base, ASSETS_DIR, prefix="cmp_baseline")
    results["baseline_q"] = evaluate_greedy(env_base, agent_base, n_episodes=eval_episodes, stage="C")

    env_double = PursuitArenaEnv(seed=42, reward_mode="full")
    agent_double = build_agent("double")
    hist_double = train(env_double, agent_double, episodes=episodes, curriculum=CURRICULUM)
    save_training_curves(hist_double, ASSETS_DIR, prefix="cmp_double")
    results["double_q"] = evaluate_greedy(env_double, agent_double, n_episodes=eval_episodes, stage="C")

    env_no_cur = PursuitArenaEnv(seed=42, reward_mode="full")
    agent_no_cur = build_agent("double")
    no_cur = [("C", episodes)]
    hist_no_cur = train(env_no_cur, agent_no_cur, episodes=episodes, curriculum=no_cur)
    save_training_curves(hist_no_cur, ASSETS_DIR, prefix="cmp_no_curriculum")
    results["no_curriculum"] = evaluate_greedy(env_no_cur, agent_no_cur, n_episodes=eval_episodes, stage="C")

    env_cur = PursuitArenaEnv(seed=42, reward_mode="full")
    agent_cur = build_agent("double")
    hist_cur = train(env_cur, agent_cur, episodes=episodes, curriculum=CURRICULUM)
    save_training_curves(hist_cur, ASSETS_DIR, prefix="cmp_curriculum")
    results["with_curriculum"] = evaluate_greedy(env_cur, agent_cur, n_episodes=eval_episodes, stage="C")

    env_simple = PursuitArenaEnv(seed=42, reward_mode="simple")
    agent_simple = build_agent("double")
    hist_simple = train(env_simple, agent_simple, episodes=episodes, curriculum=CURRICULUM)
    save_training_curves(hist_simple, ASSETS_DIR, prefix="cmp_simple_reward")
    results["simple_reward"] = evaluate_greedy(env_simple, agent_simple, n_episodes=eval_episodes, stage="C")

    env_full = PursuitArenaEnv(seed=42, reward_mode="full")
    agent_full = build_agent("double")
    hist_full = train(env_full, agent_full, episodes=episodes, curriculum=CURRICULUM)
    save_training_curves(hist_full, ASSETS_DIR, prefix="cmp_full_reward")
    results["full_reward"] = evaluate_greedy(env_full, agent_full, n_episodes=eval_episodes, stage="C")

    save_comparison_bar(results, ASSETS_DIR)
    print("Comparison metrics:")
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pursuit Arena RL")
    parser.add_argument("--mode", choices=["train", "eval", "ui", "compare"], default="train")
    parser.add_argument("--algorithm", choices=["double", "baseline"], default="double")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--stage", choices=["A", "B", "C", "MANUAL"], default="C")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--ui-mode", choices=["train_visual", "play_greedy"], default="train_visual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        run_train(save=not args.no_save, algorithm=args.algorithm, episodes=args.episodes)
    elif args.mode == "eval":
        run_eval(load=args.load, algorithm=args.algorithm, stage=args.stage, n_episodes=args.eval_episodes)
    elif args.mode == "ui":
        run_ui(ui_mode=args.ui_mode, load=args.load, algorithm=args.algorithm)
    elif args.mode == "compare":
        run_compare(episodes=args.episodes, eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
