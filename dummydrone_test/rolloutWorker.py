import tmrl.config.config_constants as cfg
import torch
from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.util import partial
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic
from environment import DUMMY_RC_DRONE_CONFIG
from tmrl.envs import GenericGymEnv

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

# Name used for training checkpoints and models saved in the TmrlData folder.
# If you change anything, also change this name (or delete the saved files in TmrlData).
my_run_name = "tutorial_minimal_drone_gpu"


# The TMRL Server is the central point of communication between TMRL entities.
# The Trainer and the RolloutWorkers connect to the Server.
security = None  # This is fine for secure local networks. On the Internet, use "TLS" instead.
password = cfg.PASSWORD  # This is the password defined in TmrlData/config/config.json

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if you want to run on the Internet.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.

# === Environment ======================================================================================================

# rtgym interface:

my_rtgym_config = DUMMY_RC_DRONE_CONFIG

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
actor_module_cls = partial(SquashedGaussianMLPActor)

# Worker local files

weights_folder = cfg.WEIGHTS_FOLDER
model_path = str(weights_folder / (my_run_name + ".tmod"))  # Current model will be stored here.
model_path_history = str(weights_folder / (my_run_name + "_"))
model_history = -1  # let us not save a model history.

# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        max_samples_per_episode=1000,
        model_path=model_path,
        # model_path_history=model_path_history,  # not used when model_history is -1
        model_history=model_history,
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet.
    # Nevertheless, it connects to the Server.

    my_worker.run(test_episode_interval=10, verbose=True) # This will make the worker collect samples and send them to the server.
