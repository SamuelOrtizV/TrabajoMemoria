# tutorial imports:
from threading import Thread

# TMRL imports:
from tmrl.networking import Trainer
from tmrl.util import partial
from tmrl.envs import GenericGymEnv
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.custom_algorithms import SpinupSacAgent
from tmrl.custom.custom_models import VanillaCNNActorCritic
from tmrl.custom.custom_memories import GenericTorchMemory

from environment import AC_Interface, AC_ENV_CONFIG
from memories import get_local_buffer_sample_imgs

# Set this to True only for debugging your pipeline.
CRC_DEBUG = True

# Name used for training checkpoints and models saved in the TmrlData folder.
# If you change anything, also change this name (or delete the saved files in TmrlData).
my_run_name = "ac_env_test3"

# === TMRL Server ======================================================================================================

# The TMRL Server is the central point of communication between TMRL entities.
# The Trainer and the RolloutWorkers connect to the Server.

security = None  # This is fine for secure local networks. On the Internet, use "TLS" instead.
password = cfg.PASSWORD  # This is the password defined in TmrlData/config/config.json

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if you want to run on the Internet.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.

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

# Trainer local files:

weights_folder = cfg.WEIGHTS_FOLDER
checkpoints_folder = cfg.CHECKPOINTS_FOLDER
model_path = str(weights_folder / (my_run_name + "_t.tmod"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))

# Dummy environment OR (observation space, action space) tuple:
# env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})
env_cls = (obs_space, act_space)

# Memory:

memory_cls = partial(cfg_obj.MEM,
                     memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                     batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                     sample_preprocessor=None,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=cfg.IMG_HIST_LEN,
                     act_buf_len=cfg.ACT_BUF_LEN,
                     crc_debug=CRC_DEBUG)

# Training agent:

training_agent_cls = cfg_obj.AGENT

# Training parameters:

epochs = 10  # maximum number of epochs, usually set this to np.inf
rounds = 10  # number of rounds per epoch
steps = 1000  # number of training steps per round
update_buffer_interval = 100
update_model_interval = 100
max_training_steps_per_env_step = 2.0
start_training = 400
device = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Training class:

training_cls = partial(
    TorchTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device)

# Trainer instance:

if __name__ == "__main__":
    my_trainer = Trainer(
        training_cls=training_cls,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        model_path=model_path,
        checkpoint_path=checkpoints_path)  # None for not saving training checkpoints
    
    my_trainer.run()

    # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:
    # my_trainer.run_with_wandb(entity=wandb_entity,
    #                           project=wandb_project,
    #                           run_id=wandb_run_id)