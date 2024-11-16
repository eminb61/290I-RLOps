from src.rl_models.sb3.sb3_contrib_local.common.maskable_recurrent.policies import (
    MaskableRecurrentActorCriticPolicy,
    MaskableRecurrentActorCriticGATPolicy
)

MlpLstmPolicy = MaskableRecurrentActorCriticPolicy
GATLstmPolicy = MaskableRecurrentActorCriticGATPolicy