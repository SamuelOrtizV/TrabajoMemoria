
# TMRL imports:
from tmrl.networking import Trainer
from tmrl.util import partial
from tmrl.envs import GenericGymEnv
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent

from environment import AC_ENV_CONFIG
from memories import MemoryFull
from custom_models import VanillaCNNActorCritic

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

# === Environment ======================================================================================================

# rtgym interface:

my_rtgym_config = AC_ENV_CONFIG

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})

# Observation and action space:

dummy_env = env_cls()
act_space = dummy_env.action_space
obs_space = dummy_env.observation_space

print(f"action space: {act_space}")
print(f"observation space: {obs_space}")

# === TMRL Trainer =====================================================================================================

# The TMRL Trainer is where your training algorithm lives.
# It connects to the Server, to retrieve training samples collected from the RolloutWorkers.
# Periodically, it also sends updated policies to the Server, which forwards them to the RolloutWorkers.

# TMRL Trainers contain a Training class. Currently, only TrainingOffline is supported.
# TrainingOffline notably contains a Memory class, and a TrainingAgent class.
# The Memory is a replay buffer. In TMRL, you are able and encouraged to define your own Memory.
# This is how you can implement highly optimized ad-hoc pipelines for your applications.
# Nevertheless, TMRL also defines a generic, non-optimized Memory that can be used for any pipeline.
# The TrainingAgent contains your training algorithm per-se.
# TrainingOffline is meant for asynchronous off-policy algorithms, such as Soft Actor-Critic.

# Dummy environment OR (observation space, action space) tuple:
# env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})
env_cls = (obs_space, act_space)

# Memory:

memory_cls = partial(MemoryFull,
                     memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                     batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                     sample_preprocessor=None,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=cfg.IMG_HIST_LEN,
                     act_buf_len=cfg.ACT_BUF_LEN,
                     crc_debug=CRC_DEBUG)

# Training agent:

#training_agent_cls = cfg_obj.AGENT

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]

if ALG_CONFIG["ALGORITHM"] == "SAC":
    training_agent_cls = partial(
            SAC_Agent,
            device='cuda' if cfg.CUDA_TRAINING else 'cpu',
            model_cls=VanillaCNNActorCritic,
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
# Implementar aqui otros algoritmos de entrenamiento si es necesario

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
    
    my_trainer.run()

    # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:
    # my_trainer.run_with_wandb(entity=wandb_entity,
    #                           project=wandb_project,
    #                           run_id=wandb_run_id)