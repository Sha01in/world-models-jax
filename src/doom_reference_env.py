"""The pinned gym-doom scenario settings on the installed ViZDoom engine."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom

from src.doom_reference import reference_preprocess


class ReferenceDoomEnv(gym.Env):
    """Preserve the legacy frame/config/action semantics without old bindings."""

    def __init__(
        self, reference_directory, asset_directory, difficulty=4, color_order="rgb"
    ):
        super().__init__()
        self.config_path = Path(reference_directory) / "legacy/take_cover.cfg"
        self.scenario_path = Path(asset_directory) / "take_cover.wad"
        self.game_path = Path(asset_directory) / "freedoom2.wad"
        for path in (self.config_path, self.scenario_path, self.game_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.game = vizdoom.DoomGame()
        self.game.load_config(str(self.config_path.resolve()))
        self.game.set_doom_scenario_path(str(self.scenario_path.resolve()))
        self.game.set_doom_game_path(str(self.game_path.resolve()))
        self.game.set_doom_map("map01")
        self.requested_difficulty = 5
        # doom-py 0.0.15 clamps setSkill(5) to4 before passing '-skill4'.
        # Modern ViZDoom accepts5, so a literal API port would change physics.
        self.effective_difficulty = difficulty
        self.game.set_doom_skill(self.effective_difficulty)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        # Old BGR24 writes R,G,B bytes; its enum name was reversed in that build.
        # Modern RGB24 preserves those effective bytes. BGR remains an ablation.
        self.effective_color_order = color_order
        self.game.set_screen_format(
            {"rgb": vizdoom.ScreenFormat.RGB24, "bgr": vizdoom.ScreenFormat.BGR24}[
                color_order
            ]
        )
        self.game.set_window_visible(False)
        self.game.set_mode(vizdoom.Mode.PLAYER)
        self.game.init()
        self.buttons = list(self.game.get_available_buttons())
        if set(self.buttons) != {vizdoom.Button.MOVE_LEFT, vizdoom.Button.MOVE_RIGHT}:
            self.game.close()
            raise ValueError("Reference scenario must have only left/right buttons")
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.steps = 0

    def observation(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros((64, 64, 3), np.uint8)
        return reference_preprocess(state.screen_buffer)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        value = float(np.asarray(action).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("Non-finite reference controller action")
        buttons = [
            int(
                value < -0.3333
                if button == vizdoom.Button.MOVE_LEFT
                else value > 0.3333
            )
            for button in self.buttons
        ]
        reward = self.game.make_action(buttons, 1)
        self.steps += 1
        dead = self.game.is_player_dead()
        finished = self.game.is_episode_finished()
        truncated = (finished or self.steps >= 2100) and not dead
        return self.observation(), reward, dead, truncated, {}

    def close(self):
        self.game.close()


def make_reference_env(
    reference_directory, asset_directory, difficulty=4, color_order="rgb"
):
    return ReferenceDoomEnv(
        reference_directory, asset_directory, difficulty, color_order
    )
