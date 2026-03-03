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
### 1) Train model (Double Q-learning)
Purpose: train the enemy AI policy with curriculum stages and save the checkpoint.
```bash
python src/main.py --mode train --algorithm double
```

When to use:
- First full training run
- Regenerating curves and model checkpoint

### 2) Evaluate model
Purpose: load a trained model and report final metrics on a target stage.
```bash
python src/main.py --mode eval --algorithm double --load --stage C
```

What it outputs:
- `capture_rate`
- `avg_steps_to_capture`
- `avg_reward`
- `invalid_move_rate`

### 3) UI (training visualization)
Purpose: watch the training process in real time (map, actions, rewards, status panel).
```bash
python src/main.py --mode ui --ui-mode train_visual --algorithm double
```

### 4) UI (greedy gameplay/demo)
Purpose: run the trained agent in gameplay mode (manual/random player mode available in UI).
```bash
python src/main.py --mode ui --ui-mode play_greedy --algorithm double --load
```

### 5) Comparison experiments (report section)
Purpose: run 3 experiment groups for analysis figures/tables:
- baseline Q-learning vs Double Q-learning
- without curriculum vs with curriculum
- simple reward vs full reward shaping

```bash
python src/main.py --mode compare --episodes 1200 --eval-episodes 120
```

### 6) Optional command variants
Train baseline Q-learning:
```bash
python src/main.py --mode train --algorithm baseline
```

Evaluate baseline model:
```bash
python src/main.py --mode eval --algorithm baseline --load --stage C
```

Quick test run (faster, lower quality):
```bash
python src/main.py --mode train --algorithm double --episodes 300
python src/main.py --mode eval --algorithm double --load --eval-episodes 60 --stage C
```

## Acceptance Targets
- Stage C capture rate >= 0.75
- Logic checks pass in train output
- Curves generated in `assets/`
