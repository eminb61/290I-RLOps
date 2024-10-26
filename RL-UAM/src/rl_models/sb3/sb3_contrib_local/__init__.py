import os

from src.rl_models.sb3.sb3_contrib_local.ppo_mask import MaskablePPO
from src.rl_models.sb3.sb3_contrib_local.ppo_recurrent import RecurrentPPO
from src.rl_models.sb3.sb3_contrib_local.ppo_mask_recurrent import MaskableRecurrentPPO

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "MaskablePPO",
    "RecurrentPPO",
    "MaskableRecurrentPPO"
]
