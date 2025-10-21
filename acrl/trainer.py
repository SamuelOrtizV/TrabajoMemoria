# TMRL imports:
from tmrl.networking import Trainer
from tmrl.util import partial
from tmrl.envs import GenericGymEnv
import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.training_offline import TorchTrainingOffline
from acrl.modules.custom_algorithms import SAC_Agent, REDQSAC_Agent

from acrl.modules.environment import AC_Interface
from acrl.modules.memories import MemoryFull
from acrl.modules.custom_models import StackedChannelCNNActorCritic, CNNRNNActorCritic, StackedChannelCNNActorCriticREDQ

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

# === Environment ======================================================================================================

CONFIG_DICT["interface"] = AC_Interface

# rtgym interface:

my_rtgym_config = CONFIG_DICT  # AC_ENV_CONFIG

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})

dummy_env = env_cls()
obs_space = dummy_env.observation_space
act_space = dummy_env.action_space

print("Observation space: ", obs_space)
print("Action space: ", act_space)

# === TMRL Trainer =====================================================================================================

# The TMRL Trainer is where the training algorithm lives.
# It connects to the Server to retrieve training samples collected from the RolloutWorkers.
# Periodically, it also sends updated policies to the Server, which forwards them to the RolloutWorkers.

# TMRL Trainers contain a Training class. Currently, we use TrainingOffline.
# TrainingOffline encapsulates a Memory class and a TrainingAgent class.
# The Memory is a replay buffer (customizable in this project).
# The TrainingAgent contains the learning algorithm (SAC or REDQ SAC here).
# TrainingOffline is intended for asynchronous off-policy algorithms such as Soft Actor-Critic.

# Dummy environment OR (observation space, action space) tuple:
# env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})
# env_cls = (obs_space, act_space)

# Memory:

memory_cls = partial(MemoryFull,
                     memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                     batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                     sample_preprocessor=None,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=cfg.IMG_HIST_LEN,
                     act_buf_len=cfg.ACT_BUF_LEN,
                     crc_debug=CRC_DEBUG)

# Model:

model_cls =  CNNRNNActorCritic if cfg.TMRL_CONFIG["USE_RNN"] else StackedChannelCNNActorCritic

# Training agent:

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]

# ALG_CONFIG["ALGORITHM"] = "REDQSAC"  # TODO: change cfg object if using REDQ outside SAC defaults

if ALG_CONFIG["ALGORITHM"] == "SAC":
    training_agent_cls = partial(
            SAC_Agent,
            device='cuda' if cfg.CUDA_TRAINING else 'cpu',
            model_cls=model_cls,
            mixed_precision=ALG_CONFIG["MIXED_PRECISION"],
            lr_actor=ALG_CONFIG["LR_ACTOR"],
            lr_critic=ALG_CONFIG["LR_CRITIC"],
            lr_entropy=ALG_CONFIG["LR_ENTROPY"],
            gamma=ALG_CONFIG["GAMMA"],
            polyak=ALG_CONFIG["POLYAK"],
            learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
            target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
            alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
            optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
            optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
            betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,
            betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,
            l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,
            l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None
        )
elif ALG_CONFIG["ALGORITHM"] == "REDQSAC":
    model_cls = StackedChannelCNNActorCriticREDQ
    training_agent_cls = partial(
        REDQSAC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=model_cls,
        mixed_precision=ALG_CONFIG["MIXED_PRECISION"],
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        n=ALG_CONFIG["REDQ_N"],  # number of Q networks
        m=ALG_CONFIG["REDQ_M"],  # number of Q targets
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
    )

print(f"Using {ALG_CONFIG['ALGORITHM']} algorithm")

# Add other training algorithms here if needed.

# Training class:

training_cls = partial(
    TorchTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
    rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
    steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
    update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
    update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
    max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
    start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"],
    device='cuda' if cfg.CUDA_TRAINING else 'cpu')

# Trainer instance:

if __name__ == "__main__":
    my_trainer = Trainer(
        training_cls=training_cls)  # None for not saving training checkpoints
    
    #my_trainer.run()

    # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:
    my_trainer.run_with_wandb()