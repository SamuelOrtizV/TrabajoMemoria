import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.networking import RolloutWorker
from tmrl.util import partial
from custom_models import SquashedGaussianVanillaCNNActor, SquashedGaussianEffNetActor
from environment import AC_Interface
from memories import get_local_buffer_sample_imgs
from tmrl.envs import GenericGymEnv
import numpy as np

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

# === Environment ======================================================================================================

CONFIG_DICT["interface"] = AC_Interface

# rtgym interface:

""" AC_ENV_CONFIG = {
    "interface": AC_Interface,  # replace this by your custom interface class
    "interface_args": (),  # arguments of your interface
    "interface_kwargs": {},  # key word arguments of your interface
    "time_step_duration": cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_duration"],  # nominal duration of your time-step
    "start_obs_capture": cfg.ENV_CONFIG["RTGYM_CONFIG"]["start_obs_capture"],  # observation retrieval will start this amount of time after the time-step begins
    # start_obs_capture should be the same as "time_step_duration" unless observation capture is non-instantaneous and
    # smaller than one time-step, and you want to capture it directly in your interface for convenience. Otherwise,
    # you need to perform observation capture in a parallel process and simply retrieve the last available observation
    # in the get_obs_rew_terminated_info() and reset() methods of your interface
    "time_step_timeout_factor": cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_timeout_factor"],  # maximum elasticity in (fraction or number of) time-steps
    "ep_max_length": cfg.ENV_CONFIG["RTGYM_CONFIG"]["ep_max_length"],  # maximum episode length
    "act_in_obs": True,  # When True, the action buffer will be appended to observations
    "act_buf_len": cfg.ACT_BUF_LEN,  # Length of the action buffer (should be max total delay + max observation capture duration, in time-steps)
    "reset_act_buf": True,  # When True, the action buffer will be filled with default actions at reset
    "benchmark": cfg.ENV_CONFIG["RTGYM_CONFIG"]["benchmark"],  # When True, a simple benchmark will be run to estimate useful timing metrics
    "benchmark_polyak": 0.1,  # Polyak averaging factor for the benchmarks (0.0 < x <= 1); smaller is slower, bigger is noisier
    "wait_on_done":cfg.ENV_CONFIG["RTGYM_CONFIG"]["wait_on_done"],  # Whether the wait() method should be called when either terminated or truncated is True
    "last_act_on_reset": False,  # When False, reset() sends the default action; when False, it sends the last action
} """

my_rtgym_config = CONFIG_DICT #AC_ENV_CONFIG

# Environment class:
env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})

# Now that we have defined our environment, let us train an agent with the generic TMRL pipeline.
# TMRL pipelines have a central communication Server, a Trainer, and one to several RolloutWorkers.


# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates its policy.

# ActorModule:

actor_module_cls = SquashedGaussianVanillaCNNActor

# SquashedGaussianMLPActor processes observations through an MLP.
# It is designed to work with the SAC algorithm.
#actor_module_cls = partial(SquashedGaussianVanillaCNNActor)

""" model_path_history = cfg.WEIGHTS_FOLDER / cfg.RUN_NAME

# check if the model path exists if not create it
if not model_path_history.exists():
    model_path_history.mkdir(parents=True, exist_ok=True) """


# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=get_local_buffer_sample_imgs,  #cfg_obj.SAMPLE_COMPRESSOR, #
        device= "cuda" if cfg.CUDA_INFERENCE else "cpu",  # True if CUDA, False if CPU (rollout worker)
        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
        standalone=True,
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        #model_path_history=model_path_history,  # not used when model_history is -1
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet.
    # Nevertheless, it connects to the Server.

    my_worker.run(test_episode_interval=20, verbose=True) # This will make the worker collect samples and send them to the server.
