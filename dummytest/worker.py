import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.networking import RolloutWorker, print_with_timestamp
from tmrl.util import partial
from custom_models import SquashedGaussianVanillaCNNActor, HumanActor,SquashedGaussianEffNetActor
from dummyenv import DUMMY_RC_DRONE_CONFIG
from memories import get_local_buffer_sample_imgs
from tmrl.envs import GenericGymEnv

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

# === Environment ======================================================================================================

# rtgym interface:

my_rtgym_config = DUMMY_RC_DRONE_CONFIG

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
    actor_module_cls = SquashedGaussianVanillaCNNActor

# SquashedGaussianMLPActor processes observations through an MLP.

# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=get_local_buffer_sample_imgs,  #cfg_obj.SAMPLE_COMPRESSOR, #
        device= "cuda" if cfg.CUDA_INFERENCE else "cpu",  # True if CUDA, False if CPU (rollout worker)
        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
        standalone=False,
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        #model_path_history=model_path_history,  # not used when model_history is -1
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet.
    # Nevertheless, it connects to the Server.

    my_worker.run(test_episode_interval=10, verbose=True, expert=cfg.TMRL_CONFIG["HUMAN_WORKER"]) # This will make the worker collect samples and send them to the server.
