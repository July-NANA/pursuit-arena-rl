# Pursuit Arena RL (CDS524 Assignment 1)

This project builds a pursuit game on a 10x10 grid with obstacles. The enemy is trained with tabular RL to catch a moving player.

## Highlights
- 10x10 action scene with obstacles and chokepoints
- Tabular Q-learning setup for pursuit behavior
- Double Q-learning (main) and baseline Q-learning (comparison)
- Curriculum stages: A (easy), B (medium), C (hard)
- Reward-shaping ablation and comparison runs
- Pygame demo with train/run/reset/player-mode toggle

## Project Structure
- `src/config.py` hyperparameters and paths
- `src/env.py` environment and state encoding
- `src/agent.py` baseline and double Q-learning
- `src/train.py` training loop
- `src/evaluate.py` greedy evaluation
- `src/metrics.py` chart generation
- `src/ui.py` pygame UI
- `src/main.py` CLI entry point
- `assets/` generated charts
- `models/` saved checkpoints
- `report/` report and demo script
- `notebook/` notebook

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run
Train double Q-learning and save the model:
```bash
python src/main.py --mode train --algorithm double
```

Evaluate a saved model on Stage C:
```bash
python src/main.py --mode eval --algorithm double --load --stage C
```

Run the UI:
```bash
python src/main.py --mode ui --ui-mode train_visual --algorithm double
python src/main.py --mode ui --ui-mode play_greedy --algorithm double --load
```

Run comparison experiments (3 groups):
```bash
python src/main.py --mode compare --episodes 1200 --eval-episodes 120
```

## Acceptance Targets
- Stage C capture rate >= 0.75
- Logic checks pass in train output
- Curves generated in `assets/`
