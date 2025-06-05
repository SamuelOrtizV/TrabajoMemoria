import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.networking import RolloutWorker, print_with_timestamp
from tmrl.util import partial
from custom_models import StackedChannelCNNActor, HumanActor, CNNRNNActor
from environment import AC_Interface
from memories import get_local_buffer_sample_imgs, MemoryFull
from tmrl.envs import GenericGymEnv

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

STANDALONE = False

# === Environment ======================================================================================================

CONFIG_DICT["interface"] = AC_Interface

my_rtgym_config = CONFIG_DICT #AC_ENV_CONFIG

# Environment class:
env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})

# Now that we have defined our environment, let us train an agent with the generic TMRL pipeline.
# TMRL pipelines have a central communication Server, a Trainer, and one to several RolloutWorkers.


# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates its policy.

import numpy as np
import itertools
import datetime
from types import SimpleNamespace

# Custom rollout worker para poder guardar los pesos de los mejores desempeños
class CustomRolloutWorker(RolloutWorker):
    def __init__(self, *args, **kwargs):
        """
        Inicializa el CustomRolloutWorker.

        Args:
            record_model_path (str): Ruta donde se guardarán los pesos del modelo cuando se rompa el récord.
        """
        super().__init__(*args, **kwargs)
        self.best_test_reward = 0.0  # Variable para rastrear el récord en pruebas
        self.weights = None
        self.infer_memory = MemoryFull(
                            memory_size=cfg.TMRL_CONFIG["IMG_STRIDE"] * cfg.IMG_HIST_LEN * 2,
                            batch_size=1,
                            imgs_obs=cfg.IMG_HIST_LEN,
                            act_buf_len=cfg.ACT_BUF_LEN,
                            device=self.device  
                            )

    def act(self, obs, test=False):
        # Empaqueta la observación como espera append_buffer (puedes poner dummy para los campos que no usas)
        dummy_sample = (
                        0,  # acción previa (dummy)
                        (obs[0], obs[1], obs[2], obs[3], *obs[4:]),  # telemetría, imágenes, acciones previas (de largo variable)
                        0.0,  # reward (dummy)
                        False,  # terminated (dummy)
                        False,  # truncated (dummy)
                        {}      # info (dummy)
                        )
    
        dummy_buffer = SimpleNamespace(memory=[dummy_sample])
        self.infer_memory.append_buffer(dummy_buffer)

        # Solo avanza el índice si hay suficientes muestras
        if len(self.infer_memory) > 0:
            last_obs, _, _, _, _, _, _ = self.infer_memory.get_transition(len(self.infer_memory) - 1)
            obs_for_model = (last_obs[0], last_obs[1], last_obs[2], last_obs[3], *last_obs[4:])
            action = self.actor.act_(obs_for_model, test=test)
        else:
            # Si no hay suficientes, usa la obs actual
            action = self.actor.act_(obs, test=test)
        return action

    def run_episode(self, max_samples=None, train=False):
        """
        Sobrescribe el método para incluir la lógica de guardar pesos al romper el récord en episodios de test.

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break

        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

        # Guardar los pesos del modelo si se rompe el récord de recompensa conseguida (solo en episodios de test)
        if not train and ret > self.best_test_reward and not self.standalone:
            self.best_test_reward = ret
            print_with_timestamp(f"\n\n\n--------NEW RECORD in test episode: {self.best_test_reward}--------\n\n\n")
            self.save_best_model()

    def save_best_model(self): #
        """
        Guarda los pesos del modelo actual cuando se rompe el récord de distancia recorrida.
        """
        if self.weights is not None:
            with open(self.model_path_history + str(self.best_test_reward) + ".tmod", 'wb') as f:
                f.write(self.weights)
            print_with_timestamp(f"Weights saved in: {self.model_path_history + str(self.best_test_reward) + ".tmod"}")
        else:
            print("NO HAY PESOS!")

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self._RolloutWorker__endpoint.receive_all(blocking=blocking) #
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            self.weights = weights #
            with open(self.model_path, 'wb') as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", 'wb') as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received


# ActorModule:
if cfg.TMRL_CONFIG["HUMAN_WORKER"]:
    actor_module_cls = HumanActor
else:
    actor_module_cls = StackedChannelCNNActor #CNNRNNActor #

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

    # Note: at this point, the RolloutWorker is not collecting samples yet.
    # Nevertheless, it connects to the Server.

    if STANDALONE:
        # In standalone mode, the worker will not connect to a server.
        my_worker.run_episodes(max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE)
    else:
        my_worker.run(test_episode_interval=10, verbose=True, expert=cfg.TMRL_CONFIG["HUMAN_WORKER"]) # This will make the worker collect samples and send them to the server.