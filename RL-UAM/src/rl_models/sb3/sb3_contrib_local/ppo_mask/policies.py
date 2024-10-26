from sb3_contrib.common.maskable.policies import (
    MaskableActorCriticCnnPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)

from src.rl_models.sb3.sb3_contrib_local.common.maskable.policies import MaskableActorCriticGATPolicy

MlpPolicy = MaskableActorCriticPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy
GATPolicy = MaskableActorCriticGATPolicy
