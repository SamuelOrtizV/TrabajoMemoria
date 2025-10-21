import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.networking import print_with_timestamp
from tmrl.util import partial
from modules.custom_models import StackedChannelCNNActor, HumanActor, CNNRNNActor
from modules.environment import AC_Interface
from modules.memories import get_local_buffer_sample_imgs
from tmrl.envs import GenericGymEnv
from modules.custom_worker import CustomRolloutWorker

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

STANDALONE = True

# === Environment ======================================================================================================

CONFIG_DICT["interface"] = AC_Interface

my_rtgym_config = CONFIG_DICT #AC_ENV_CONFIG

# Environment class:
env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})

# Now that the environment is defined, we set up the TMRL pipeline side for the worker.
# TMRL pipelines have a central communication Server, a Trainer, and one to several RolloutWorkers.


# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates its policy.

from types import SimpleNamespace
# Custom rollout worker: saves best-performing weights and supports expert mode.

# ActorModule:
if cfg.TMRL_CONFIG["HUMAN_WORKER"]:
    actor_module_cls = HumanActor
else:
    actor_module_cls = CNNRNNActor if cfg.TMRL_CONFIG["USE_RNN"] else StackedChannelCNNActor

# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = CustomRolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=get_local_buffer_sample_imgs,  #cfg_obj.SAMPLE_COMPRESSOR, #
        device= "cuda" if cfg.CUDA_INFERENCE else "cpu",  # True if CUDA, False if CPU (rollout worker)
        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
        standalone=STANDALONE,
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet,
    # but it connects to the Server.

    if STANDALONE:
        # In standalone mode, the worker will not connect to a server.
        print_with_timestamp("Running in standalone mode, no server connection.")
        my_worker.run_episodes(max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE)
    else:
        # This will make the worker collect samples and send them to the server.
        my_worker.run(test_episode_interval=10, verbose=True, expert=cfg.TMRL_CONFIG["HUMAN_WORKER"])