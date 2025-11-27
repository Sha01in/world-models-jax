import gymnasium as gym
import numpy as np
import cv2
import os
from typing import Tuple

try:
    import vizdoom
    from vizdoom import DoomGame, ScreenFormat, ScreenResolution
except ImportError:
    vizdoom = None

class VizDoomEnv(gym.Env):
    def __init__(self, env_name, render_mode=None, img_size=64):
        super().__init__()
        if vizdoom is None:
            raise ImportError("vizdoom is not installed. Please install it via 'pip install vizdoom'")
            
        self.img_size = img_size
        self.render_mode = render_mode
        
        # Setup DoomGame
        self.game = DoomGame()
        
        # Find config path
        scenarios_path = vizdoom.scenarios_path
        if "TakeCover" in env_name:
            config_path = os.path.join(scenarios_path, "take_cover.cfg")
        else:
            # Fallback or error
            raise ValueError(f"Unsupported Doom environment: {env_name}")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Doom config not found at {config_path}")
            
        self.game.load_config(config_path)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        
        if render_mode == "human":
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)
            
        self.game.init()
        
        # Action Space: Move Left, Move Right
        # TakeCover usually has MOVE_LEFT and MOVE_RIGHT buttons available
        self.action_space = gym.spaces.Discrete(2)
        
        # Observation Space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        return self._get_obs(), {}
        
    def step(self, action):
        # Action mapping: 0 -> Left, 1 -> Right
        # Buttons in TakeCover: [MOVE_LEFT, MOVE_RIGHT]
        # We need to check the available buttons in the cfg, but usually it's standard.
        # Let's assume [MOVE_LEFT, MOVE_RIGHT] are the first two buttons if defined.
        # Or we construct the action list explicitly.
        
        actions = [0, 0]
        # Continuous to Discrete Mapping
        # action is typically a numpy array or float
        if isinstance(action, (np.ndarray, list)):
            val = action[0]
        else:
            val = action
            
        if val < -0.3:
            actions[0] = 1 # Move Left
        elif val > 0.3:
            actions[1] = 1 # Move Right
        # Else: No-Op (Wait)
            
        reward = self.game.make_action(actions)
        
        terminated = self.game.is_episode_finished()
        truncated = False # Doom episodes usually end by death or timeout handled by game
        
        if terminated:
            try:
                health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
                print(f"[INFO] Episode Terminated. Health: {health}")
            except Exception:
                pass
            obs = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            obs = self._get_obs()
            
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        state = self.game.get_state()
        if state is not None:
            img = state.screen_buffer
            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            return img
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def close(self):
        self.game.close()

def make_env(env_name: str, render_mode=None):
    if "CarRacing" in env_name:
        env = gym.make(env_name, render_mode=render_mode)
    elif "Doom" in env_name or "TakeCover" in env_name:
        # Use our custom wrapper
        env = VizDoomEnv(env_name, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
        
    return env
