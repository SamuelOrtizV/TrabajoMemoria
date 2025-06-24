import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.networking import RolloutWorker, print_with_timestamp
from tmrl.util import partial
from custom_models import StackedChannelCNNActor, HumanActor, CNNRNNActor
from environment import AC_Interface
from memories import MemoryInference, get_local_buffer_sample_imgs
from tmrl.envs import GenericGymEnv
from util import ImageVisualizer

import cv2

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
import os
import itertools
import datetime
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
# Custom rollout worker para poder guardar los pesos de los mejores desempeños

class CustomRolloutWorker(RolloutWorker):
    def __init__(self, *args, **kwargs):
        """
        Inicializa el CustomRolloutWorker.

        Args:
            record_model_path (str): Ruta donde se guardarán los pesos del modelo cuando se rompa el récord.
        """
        super().__init__(*args, **kwargs)
        self.weights_folder = cfg.WEIGHTS_FOLDER
        self.run_name = cfg.RUN_NAME
        self.best_test_reward = self.init_best_test_reward()  # Variable para rastrear el récord en pruebas
        self.weights = None
        self.view_input = cfg.TMRL_CONFIG["VIEW_INPUT_TENSOR"] #TODO agregarlo al cfg
        self.visualizer = ImageVisualizer(title="Input tensor visualization")
        log_dir = "runs/" + self.run_name
        train_tag = "train/episode_reward"
        test_tag = "test/episode_reward"
        self.episode_counter_train = self.get_last_step(log_dir, train_tag)
        self.episode_counter_test = self.get_last_step(log_dir, test_tag)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.stride = cfg.TMRL_CONFIG["IMG_STRIDE"]
        self.img_hist_len = cfg.IMG_HIST_LEN
        self.act_buf_len = cfg.ACT_BUF_LEN
        self.infer_memory = MemoryInference(
                            capacity= self.img_hist_len* self.stride*2,
                            hist_len=self.img_hist_len,
                            act_len= self.act_buf_len,
                            stride=self.stride)

    def get_last_step(self, log_dir, tag): #tal vez se puede sacar de la clase 
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        if not os.path.exists(log_dir):
            return 0  # La carpeta no existe, así que no hay episodios previos

        event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
        if not event_files:
            return 0  # No hay archivos de eventos

        event_file = max([os.path.join(log_dir, f) for f in event_files], key=os.path.getctime)
        ea = EventAccumulator(event_file)
        ea.Reload()
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            if events:
                print_with_timestamp(f"Last step for tag '{tag}': {events[-1].step + 1}")
                return events[-1].step + 1  # Siguiente episodio
            
        episodes = input(f"Could not load last step for tag '{tag}'. Enter the number of episodes to start from: ")

        return episodes if episodes.isdigit() else 0  # Si no es un número, empieza desde 0
    
    def init_best_test_reward(self):
          # --- Buscar archivos de pesos con "rec" en el nombre ---
        if hasattr(self, "model_path_history"):
            files = [f for f in os.listdir(self.weights_folder) if "rec" in f and self.run_name in f and f.endswith(".tmod")]
            if files:
                # Extraer el número de recompensa de cada archivo
                rewards = []
                for fname in files:
                    # Ejemplo: rec_123.45.tmod
                    try:
                        num_str = fname.split("_")[-1].replace(".tmod", "")
                        reward = float(num_str)
                        rewards.append(reward)
                    except Exception:
                        continue
                if rewards:
                    print_with_timestamp(f"Best test reward found in history: {max(rewards)}")
                    return max(rewards)
        return 0.0
                    
    def act(self, obs, test=False):
        if self.stride > 1:            
            self.infer_memory.append(obs)

            if len(self.infer_memory) > self.img_hist_len * self.stride:
                
                obs_for_model = self.infer_memory.get_transition()

                action = self.actor.act_(obs_for_model, test=test)

                if self.view_input:
                    self.view_input_tensor(obs_for_model)
            else:
                action = self.actor.act_(obs, test=test)
            return action
        else:
            # Usa el método original de la clase base
            return super().act(obs, test=test)

    def collect_train_episode(self, max_samples=None):
        self.env.env.env._RealTimeEnvTS__interface.train_mode = True
        super().collect_train_episode(max_samples=max_samples)
        if hasattr(self, "tb_writer"):
            self.tb_writer.add_scalar("train/episode_reward", self.buffer.stat_train_return, self.episode_counter_train)
            self.tb_writer.add_scalar("train/episode_length",  self.buffer.stat_train_steps, self.episode_counter_train)
            self.episode_counter_train += 1
            
    def run_episode(self, max_samples=None, train=False):
        """
        Metodo para episodios de prueba
        Sobrescribe el método para incluir la lógica de guardar pesos al romper el récord en episodios de test.

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        
        self.env.env.env._RealTimeEnvTS__interface.train_mode = train  # Esta super anidado...

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

        if hasattr(self, "tb_writer"):
            self.tb_writer.add_scalar("test/episode_reward", ret, self.episode_counter_test)
            self.tb_writer.add_scalar("test/episode_length", steps, self.episode_counter_test)
            self.episode_counter_test += 1

        # Guardar los pesos del modelo si se rompe el récord de recompensa conseguida (solo en episodios de test)
        if not train and ret > self.best_test_reward and not self.standalone:
            self.best_test_reward = ret
            print_with_timestamp(f"\n\n\n--------NEW RECORD in test episode: {self.best_test_reward}--------\n\n\n")
            self.save_best_model()
        print("\n")

    def save_best_model(self): #
        """
        Guarda los pesos del modelo actual cuando se rompe el récord de distancia recorrida.
        """
        if self.weights is not None:
            path = self.model_path_history +"rec_"+ str(self.best_test_reward) + ".tmod"
            with open(path, 'wb') as f:
                f.write(self.weights)
            print_with_timestamp(f"Weights saved in: {path}")
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

    def view_input_tensor(self, x):
        """
        Visualiza la información pasada al agente usando Tkinter.
        Las imágenes históricas se concatenan horizontalmente (más antigua a la izquierda).
        El título se imprime por consola.
        """

        """ print("\n--- [DEBUG] Tipos de entrada en forward ---")
        for i, item in enumerate(x):
            print(f"x[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}") """

        # Desempaquetar los datos del batch
        speed, gear, rpm, images, *prev_act = x

        # Seleccionar el i-ésimo elemento de cada uno
        #TODO DESNORMALIZAR ESTOS VALORES
        max_speed = 400
        max_gear = 10
        max_rpm = 20000

        s_val = speed.item() * max_speed #TODO revisar si es necesario usar el item() aquí
        g_val = gear.item() * max_gear
        r_val = rpm.item() * max_rpm
        acts_vals = [pa for pa in prev_act]

        # Armar string para el título
        acts_str = ", ".join([f"Prev Act{j+1}: {val}" for j, val in enumerate(acts_vals)])
        title_str = f"Speed: {s_val:.0f}, Gear: {g_val:.0f}, RPM: {r_val:.0f}, {acts_str}"
        #print(title_str)

        # Extraer imágenes
        img_tensor = images  # shape: (num_imgs, H, W, C)

        # Convertir a numpy y uint8 si es necesario
        imgs = img_tensor
        if imgs.dtype != np.uint8:
            imgs = (imgs * 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)

        # Convertir a lista de imágenes individuales
        img_list = [img for img in imgs]  # Cada img: (H, W, C)

        # Concatenar horizontalmente usando OpenCV
        concat_img = cv2.hconcat(img_list)  # (H, num_imgs*W, C)

        # Visualización con OpenCV (convertir a BGR si es necesario)
        cv2.imshow("Input tensor visualization", concat_img[..., ::-1])
        cv2.waitKey(1)

# ActorModule:
if cfg.TMRL_CONFIG["HUMAN_WORKER"]:
    actor_module_cls = HumanActor
else:
    actor_module_cls = StackedChannelCNNActor # CNNRNNActor #

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