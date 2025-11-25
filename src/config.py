from dataclasses import dataclass
from typing import Tuple

@dataclass
class EnvConfig:
    env_name: str
    latent_dim: int
    hidden_size: int
    action_dim: int
    action_range: Tuple[float, float] = (-1.0, 1.0) # For continuous actions
    
    # Training params
    vae_batch_size: int = 32
    rnn_batch_size: int = 32
    
    # Doom specific
    is_doom: bool = False

# CarRacing-v3 Configuration
carracing_config = EnvConfig(
    env_name="CarRacing-v3",
    latent_dim=32,
    hidden_size=256,
    action_dim=3,
    is_doom=False
)

# VizDoom Take Cover Configuration
doom_config = EnvConfig(
    env_name="VizdoomTakeCover-v0", # Using shimmy/gymnasium naming if available, or we'll wrap it
    latent_dim=64,
    hidden_size=512,
    action_dim=1, # We'll map continuous output to discrete move left/right/wait
    is_doom=True
)

def get_config(env_name: str) -> EnvConfig:
    if "CarRacing" in env_name:
        return carracing_config
    elif "Doom" in env_name or "TakeCover" in env_name:
        return doom_config
    else:
        raise ValueError(f"Unknown environment: {env_name}")
