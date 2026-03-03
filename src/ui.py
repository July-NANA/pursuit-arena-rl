from __future__ import annotations

import time
from dataclasses import dataclass

import pygame

from config import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_TO_ARROW,
    ACTION_UP,
    CELL_SIZE,
    COLOR_BG,
    COLOR_BTN,
    COLOR_BTN_TEXT,
    COLOR_CAPTURE,
    COLOR_ENEMY,
    COLOR_GRID,
    COLOR_OBSTACLE,
    COLOR_PLAYER,
    COLOR_TEXT,
    COLOR_TIMEOUT,
    CURRICULUM,
    FPS,
    GRID_COLS,
    GRID_ROWS,
    PADDING,
    PANEL_WIDTH,
    TRAIN_EPISODES,
)
from train import train


@dataclass
class UIButton:
    label: str
    rect: pygame.Rect

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        pygame.draw.rect(screen, COLOR_BTN, self.rect, border_radius=8)
        text = font.render(self.label, True, COLOR_BTN_TEXT)
        screen.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class PursuitArenaUI:
    def __init__(self, env, agent) -> None:
        pygame.init()
        pygame.display.set_caption("Pursuit Arena RL")

        self.env = env
        self.agent = agent
        self.running = True
        self.player_mode = "random"

        self.grid_w = GRID_COLS * CELL_SIZE
        self.grid_h = GRID_ROWS * CELL_SIZE
        self.width = self.grid_w + PANEL_WIDTH + PADDING * 3
        self.height = self.grid_h + PADDING * 2

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.small = pygame.font.SysFont("Arial", 16)

        panel_x = PADDING * 2 + self.grid_w
        self.btn_train = UIButton("Train", pygame.Rect(panel_x, 56, 240, 42))
        self.btn_run = UIButton("Run Greedy", pygame.Rect(panel_x, 108, 240, 42))
        self.btn_reset = UIButton("Reset", pygame.Rect(panel_x, 160, 240, 42))
        self.btn_toggle = UIButton("Toggle Player Mode", pygame.Rect(panel_x, 212, 240, 42))

        self.status = {
            "mode": "idle",
            "episode": 0,
            "stage": "-",
            "step": 0,
            "reward": 0.0,
            "epsilon": self.agent.epsilon,
            "distance": 0,
            "los": 0,
            "event": "",
            "capture": False,
            "action": None,
        }

    def _draw_grid(self) -> None:
        data = self.env.render_data()
        ox, oy = PADDING, PADDING

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rect = pygame.Rect(ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = COLOR_BG
                if (r, c) in data["obstacles"]:
                    color = COLOR_OBSTACLE
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)

        pr, pc = data["player"]
        er, ec = data["enemy"]

        player_rect = pygame.Rect(ox + pc * CELL_SIZE + 8, oy + pr * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16)
        enemy_rect = pygame.Rect(ox + ec * CELL_SIZE + 8, oy + er * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16)
        pygame.draw.ellipse(self.screen, COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, COLOR_ENEMY, enemy_rect, border_radius=6)

    def _draw_panel(self) -> None:
        panel_x = PADDING * 2 + self.grid_w
        title = self.font.render("Pursuit Arena", True, COLOR_TEXT)
        self.screen.blit(title, (panel_x, 16))

        self.btn_train.draw(self.screen, self.font)
        self.btn_run.draw(self.screen, self.font)
        self.btn_reset.draw(self.screen, self.font)
        self.btn_toggle.draw(self.screen, self.small)

        event_color = COLOR_CAPTURE if self.status["event"] == "capture" else COLOR_TIMEOUT if self.status["event"] == "timeout" else COLOR_TEXT

        lines = [
            f"Mode: {self.status['mode']}",
            f"Player mode: {self.player_mode}",
            f"Episode: {self.status['episode']}",
            f"Stage: {self.status['stage']}",
            f"Step: {self.status['step']}",
            f"Reward: {self.status['reward']:.2f}",
            f"Epsilon: {self.status['epsilon']:.3f}",
            f"Distance: {self.status['distance']}",
            f"Line of sight: {self.status['los']}",
            f"AI action: {self.status['action']}",
            f"Capture: {self.status['capture']}",
        ]

        y = 280
        for line in lines:
            t = self.small.render(line, True, COLOR_TEXT)
            self.screen.blit(t, (panel_x, y))
            y += 22

        evt = self.small.render(f"Event: {self.status['event']}", True, event_color)
        self.screen.blit(evt, (panel_x, y + 8))

        hint = [
            "Keys:",
            "T=Train, G=Run, R=Reset, M=Toggle",
            "WASD/Arrows to move player in manual mode",
            "ESC=Exit",
        ]
        y = self.height - 88
        for line in hint:
            h = self.small.render(line, True, COLOR_TEXT)
            self.screen.blit(h, (panel_x, y))
            y += 20

    def _render(self) -> None:
        self.screen.fill(COLOR_BG)
        self._draw_grid()
        self._draw_panel()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _set_manual_action_from_key(self, key: int) -> None:
        action = None
        if key in (pygame.K_w, pygame.K_UP):
            action = ACTION_UP
        elif key in (pygame.K_s, pygame.K_DOWN):
            action = ACTION_DOWN
        elif key in (pygame.K_a, pygame.K_LEFT):
            action = ACTION_LEFT
        elif key in (pygame.K_d, pygame.K_RIGHT):
            action = ACTION_RIGHT

        if action is None:
            return

        if self.status.get("mode") == "play_greedy":
            self.env.set_manual_player_action(action)
            return

        moved = self.env.manual_move_player(action)
        self.env.set_manual_player_action(-1)

        if moved:
            state = self.env.encode_state()
            ai_action = self.agent.select_action(state, training=False)
            _, reward, done, info = self.env.step(ai_action)
            self.status.update(
                {
                    "mode": "manual_turn",
                    "step": self.status.get("step", 0) + 1,
                    "reward": self.status.get("reward", 0.0) + reward,
                    "distance": info["distance"],
                    "los": info["line_of_sight"],
                    "event": info["event"],
                    "capture": info["event"] == "capture",
                    "action": ACTION_TO_ARROW[ai_action],
                }
            )
            if done:
                self.status["mode"] = "manual_done"
        else:
            enemy = self.env.enemy_pos
            player = self.env.player_pos
            distance = abs(enemy[0] - player[0]) + abs(enemy[1] - player[1])
            los = self.env._line_of_sight(enemy, player)
            self.status.update(
                {
                    "event": "manual_blocked",
                    "distance": distance,
                    "los": los,
                }
            )

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_t:
                    self.start_train_visual()
                elif event.key == pygame.K_g:
                    self.play_greedy()
                elif event.key == pygame.K_r:
                    self.reset_scene()
                elif event.key == pygame.K_m:
                    self.toggle_player_mode()
                elif self.player_mode == "manual":
                    self._set_manual_action_from_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_train.hit(event.pos):
                    self.start_train_visual()
                elif self.btn_run.hit(event.pos):
                    self.play_greedy()
                elif self.btn_reset.hit(event.pos):
                    self.reset_scene()
                elif self.btn_toggle.hit(event.pos):
                    self.toggle_player_mode()

    def _train_cb(self, payload: dict) -> None:
        self.status.update(
            {
                "mode": "train_visual",
                "episode": payload["episode"],
                "stage": payload["stage"],
                "step": payload["step"],
                "reward": payload["total_reward"],
                "epsilon": payload["epsilon"],
                "distance": payload["distance"],
                "los": payload["line_of_sight"],
                "event": payload["event"],
                "capture": payload["capture"],
                "action": ACTION_TO_ARROW.get(payload["action"], "-") if payload["action"] is not None else "-",
            }
        )
        self._events()
        self._render()

    def start_train_visual(self) -> None:
        self.status["mode"] = "train_visual"
        train(self.env, self.agent, episodes=TRAIN_EPISODES, curriculum=CURRICULUM, callback=self._train_cb)

    def play_greedy(self) -> None:
        stage = "MANUAL" if self.player_mode == "manual" else "C"
        if self.env.stage != stage or self.status.get("mode") == "idle":
            state = self.env.reset(stage=stage)
        else:
            state = self.env.encode_state()
        done = False
        steps = int(self.status.get("step", 0))
        reward_sum = float(self.status.get("reward", 0.0))

        while not done and self.running:
            self._events()
            action = self.agent.select_action(state, training=False)
            nxt, reward, done, info = self.env.step(action)
            state = nxt
            steps += 1
            reward_sum += reward
            self.status.update(
                {
                    "mode": "play_greedy",
                    "episode": 1,
                    "stage": stage,
                    "step": steps,
                    "reward": reward_sum,
                    "epsilon": self.agent.epsilon,
                    "distance": info["distance"],
                    "los": info["line_of_sight"],
                    "event": info["event"],
                    "capture": info["event"] == "capture",
                    "action": ACTION_TO_ARROW[action],
                }
            )
            self._render()
            time.sleep(0.08)

    def toggle_player_mode(self) -> None:
        self.player_mode = "manual" if self.player_mode == "random" else "random"
        self.reset_scene()

    def reset_scene(self) -> None:
        stage = "MANUAL" if self.player_mode == "manual" else "C"
        self.env.reset(stage=stage)
        self.status.update(
            {
                "mode": "reset",
                "episode": 0,
                "stage": stage,
                "step": 0,
                "reward": 0.0,
                "distance": 0,
                "event": "reset",
                "capture": False,
                "action": None,
            }
        )

    def run(self, mode: str) -> None:
        if mode == "play_greedy":
            self.play_greedy()

        while self.running:
            self._events()
            self._render()

        pygame.quit()


def run_demo(env, agent, mode: str) -> None:
    if mode not in {"train_visual", "play_greedy"}:
        raise ValueError("mode must be one of {'train_visual', 'play_greedy'}")
    app = PursuitArenaUI(env, agent)
    app.run(mode=mode)
