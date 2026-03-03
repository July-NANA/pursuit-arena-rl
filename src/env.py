from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from config import (
    ACTIONS,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_TO_DELTA,
    ACTION_UP,
    ADJ_MASK_SIZE,
    DIST_BUCKET_MAX,
    DIST_BUCKET_MIN,
    DIST_BUCKET_SIZE,
    GRID_COLS,
    GRID_ROWS,
    MAX_STEPS,
    N_STATES,
    OBSTACLES,
    PLAYER_DIR_SIZE,
    RANDOM_SEED,
    REWARD_CAPTURE,
    REWARD_CLOSER,
    REWARD_DANGER,
    REWARD_FARTHER,
    REWARD_INVALID,
    REWARD_SAME_DIST,
    REWARD_STEP_COST,
)

Position = Tuple[int, int]


@dataclass
class PursuitArenaEnv:
    rows: int = GRID_ROWS
    cols: int = GRID_COLS
    max_steps: int = MAX_STEPS
    seed: int = RANDOM_SEED
    reward_mode: str = "full"

    def __post_init__(self) -> None:
        self.obstacles = set(OBSTACLES)
        self.rng = np.random.default_rng(self.seed)
        self.n_states = N_STATES
        self.n_actions = len(ACTIONS)
        self.enemy_pos: Position = (0, 0)
        self.player_pos: Position = (self.rows - 1, self.cols - 1)
        self.player_recent_dir = 4
        self.stage = "B"
        self.steps = 0
        self.invalid_streak = 0
        self.manual_player_action = -1

    def _in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_free(self, pos: Position) -> bool:
        return self._in_bounds(pos) and pos not in self.obstacles

    def _move(self, pos: Position, action: int) -> Position:
        dr, dc = ACTION_TO_DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if self._is_free(nxt):
            return nxt
        return pos

    def _manhattan(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _sample_free_cell(self) -> Position:
        while True:
            cell = (int(self.rng.integers(0, self.rows)), int(self.rng.integers(0, self.cols)))
            if cell not in self.obstacles:
                return cell

    def _sample_start_pair(self, min_dist: int) -> tuple[Position, Position]:
        for _ in range(500):
            enemy = self._sample_free_cell()
            player = self._sample_free_cell()
            if enemy != player and self._manhattan(enemy, player) >= min_dist:
                return enemy, player
        return (0, 0), (self.rows - 1, self.cols - 1)

    def _line_of_sight(self, enemy: Position, player: Position) -> int:
        if enemy[0] == player[0]:
            row = enemy[0]
            c1, c2 = sorted([enemy[1], player[1]])
            for c in range(c1 + 1, c2):
                if (row, c) in self.obstacles:
                    return 0
            return 1
        if enemy[1] == player[1]:
            col = enemy[1]
            r1, r2 = sorted([enemy[0], player[0]])
            for r in range(r1 + 1, r2):
                if (r, col) in self.obstacles:
                    return 0
            return 1
        return 0

    def _adjacent_obstacle_mask(self, pos: Position) -> int:
        mask = 0
        candidates = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
        for idx, action in enumerate(candidates):
            dr, dc = ACTION_TO_DELTA[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            blocked = not self._is_free(nxt)
            if blocked:
                mask |= (1 << idx)
        return mask

    def _danger_flag(self, pos: Position) -> int:
        mask = self._adjacent_obstacle_mask(pos)
        blocked_count = int(mask & 1 > 0) + int(mask & 2 > 0) + int(mask & 4 > 0) + int(mask & 8 > 0)
        near_boundary = pos[0] in (0, self.rows - 1) or pos[1] in (0, self.cols - 1)
        if blocked_count >= 3:
            return 1
        if blocked_count >= 2 and near_boundary:
            return 1
        return 0

    def _bucket_dist(self, d: int) -> int:
        clipped = min(max(d, DIST_BUCKET_MIN), DIST_BUCKET_MAX)
        return clipped - DIST_BUCKET_MIN

    def encode_state(self) -> int:
        dx = self.player_pos[0] - self.enemy_pos[0]
        dy = self.player_pos[1] - self.enemy_pos[1]
        dx_b = self._bucket_dist(dx)
        dy_b = self._bucket_dist(dy)
        los = self._line_of_sight(self.enemy_pos, self.player_pos)
        adj_mask = self._adjacent_obstacle_mask(self.enemy_pos)
        pdir = self.player_recent_dir
        danger = self._danger_flag(self.enemy_pos)

        sid = dx_b
        sid = sid * DIST_BUCKET_SIZE + dy_b
        sid = sid * 2 + los
        sid = sid * ADJ_MASK_SIZE + adj_mask
        sid = sid * PLAYER_DIR_SIZE + pdir
        sid = sid * 2 + danger
        return int(sid)

    def _player_policy_action(self, stage: str) -> int:
        if stage == "MANUAL":
            action = self.manual_player_action
            self.manual_player_action = -1
            if action in ACTIONS and self._move(self.player_pos, action) != self.player_pos:
                self.player_recent_dir = action
                return action
            self.player_recent_dir = 4
            return -1

        legal_actions = [a for a in ACTIONS if self._move(self.player_pos, a) != self.player_pos]
        if not legal_actions:
            self.player_recent_dir = 4
            return -1

        if stage == "A":
            if self.rng.random() < 0.6:
                self.player_recent_dir = 4
                return -1
            action = int(self.rng.choice(legal_actions))
            self.player_recent_dir = action
            return action

        if stage == "B":
            action = int(self.rng.choice(legal_actions))
            self.player_recent_dir = action
            return action

        best_actions = []
        best_score = -10**9
        for action in legal_actions:
            nxt = self._move(self.player_pos, action)
            score = self._manhattan(nxt, self.enemy_pos)
            if score > best_score:
                best_actions = [action]
                best_score = score
            elif score == best_score:
                best_actions.append(action)
        if self.rng.random() < 0.7:
            action = int(self.rng.choice(best_actions))
        else:
            action = int(self.rng.choice(legal_actions))
        self.player_recent_dir = action
        return action

    def reset(self, stage: str = "B") -> int:
        self.stage = stage
        self.steps = 0
        self.invalid_streak = 0
        self.player_recent_dir = 4
        self.manual_player_action = -1

        if stage == "A":
            self.enemy_pos, self.player_pos = self._sample_start_pair(min_dist=4)
        elif stage == "B":
            self.enemy_pos, self.player_pos = self._sample_start_pair(min_dist=6)
        elif stage == "MANUAL":
            self.enemy_pos, self.player_pos = self._sample_start_pair(min_dist=6)
        else:
            self.enemy_pos, self.player_pos = self._sample_start_pair(min_dist=7)

        return self.encode_state()

    def set_manual_player_action(self, action: int) -> None:
        self.manual_player_action = action

    def manual_move_player(self, action: int) -> bool:
        """Move player immediately in manual mode (without advancing enemy/episode)."""
        if action not in ACTIONS:
            return False
        nxt = self._move(self.player_pos, action)
        moved = nxt != self.player_pos
        self.player_pos = nxt
        self.player_recent_dir = action if moved else 4
        return moved

    def step(self, action: int) -> tuple[int, float, bool, Dict[str, object]]:
        self.steps += 1
        prev_dist = self._manhattan(self.enemy_pos, self.player_pos)

        proposed_enemy = self._move(self.enemy_pos, action)
        invalid_move = proposed_enemy == self.enemy_pos
        if invalid_move:
            self.invalid_streak += 1
        else:
            self.invalid_streak = 0
        self.enemy_pos = proposed_enemy

        if self._manhattan(self.enemy_pos, self.player_pos) <= 1:
            info = {
                "event": "capture",
                "distance": self._manhattan(self.enemy_pos, self.player_pos),
                "line_of_sight": self._line_of_sight(self.enemy_pos, self.player_pos),
                "invalid_move": invalid_move,
            }
            return self.encode_state(), REWARD_CAPTURE, True, info

        player_action = self._player_policy_action(self.stage)
        if player_action != -1:
            self.player_pos = self._move(self.player_pos, player_action)

        now_dist = self._manhattan(self.enemy_pos, self.player_pos)
        done = False
        event = "step"

        if self._manhattan(self.enemy_pos, self.player_pos) <= 1:
            done = True
            event = "capture"
            reward = REWARD_CAPTURE
        elif self.steps >= self.max_steps:
            done = True
            event = "timeout"
            reward = REWARD_STEP_COST
        else:
            reward = REWARD_STEP_COST
            if invalid_move:
                reward += REWARD_INVALID
            if not invalid_move and self.reward_mode == "full":
                if now_dist < prev_dist:
                    reward += REWARD_CLOSER
                elif now_dist > prev_dist:
                    reward += REWARD_FARTHER
                else:
                    reward += REWARD_SAME_DIST
                if self._danger_flag(self.enemy_pos) == 1:
                    reward += REWARD_DANGER

        info = {
            "event": event if not invalid_move else "invalid" if event == "step" else event,
            "distance": now_dist,
            "line_of_sight": self._line_of_sight(self.enemy_pos, self.player_pos),
            "invalid_move": invalid_move,
        }
        return self.encode_state(), float(reward), done, info

    def render_data(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "obstacles": set(self.obstacles),
            "enemy": self.enemy_pos,
            "player": self.player_pos,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "stage": self.stage,
        }
