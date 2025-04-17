import tmrl.config.config_constants as cfg
from tmrl.networking import RolloutWorker
from tmrl.util import partial
from custom_models import SquashedGaussianVanillaCNNActor
from environment import AC_ENV_CONFIG
from memories import get_local_buffer_sample_imgs
from tmrl.envs import GenericGymEnv

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


# Now that we have defined our environment, let us train an agent with the generic TMRL pipeline.
# TMRL pipelines have a central communication Server, a Trainer, and one to several RolloutWorkers.


# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates its policy.

# ActorModule:

# SquashedGaussianMLPActor processes observations through an MLP.
# It is designed to work with the SAC algorithm.
#actor_module_cls = partial(SquashedGaussianVanillaCNNActor)

model_history = -1  # let us not save a model history.

# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=SquashedGaussianVanillaCNNActor,
        sample_compressor=get_local_buffer_sample_imgs,  #cfg_obj.SAMPLE_COMPRESSOR, #
        device= "cuda" if cfg.CUDA_INFERENCE else "cpu",  # True if CUDA, False if CPU (rollout worker)
        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
        # model_path_history=model_path_history,  # not used when model_history is -1
        model_history=model_history,
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet.
    # Nevertheless, it connects to the Server.

    my_worker.run(test_episode_interval=10, verbose=True) # This will make the worker collect samples and send them to the server.
