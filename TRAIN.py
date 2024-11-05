import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import *
import os
import torchvision.transforms.functional as F
from inputs.xbox_controller_emulator import XboxControllerEmulator
from inputs.GameInputs import reset_environment
from inputs.getkeys import key_check
from UDP_listener import udp_listener
from ScreenRecorder import *
from torchvision import transforms
import time
import torch.nn.functional as F
import threading

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Eventos para detener y pausar el hilo
stop_event = threading.Event()
pause_event = threading.Event()

def key_detection(): 
    global stop_event, pause_event 
    while not stop_event.is_set():
        keys = key_check()
        if keys == "Q":
            stop_event.set()
        elif keys == "P":
            pause_event.clear() if pause_event.is_set() else pause_event.set()
            print(f"{'Reanudando' if not pause_event.is_set() else 'Pausando'} el modelo...                                                          ", end="\r")
            time.sleep(1)  # Evitar múltiples detecciones rápidas

# Hilo de detección de teclas
key_thread = threading.Thread(target=key_detection)
key_thread.start()  

# Parámetros del entorno
screen_size = (1920, 1080)
fps = 100  # Ajuste de FPS según capacidad del sistema
full_screen = True
input_size = (240, 135)  # Tamaño de entrada de la imagen
save_dir = "./trained_models/SACtest"
os.makedirs(save_dir, exist_ok=True)

# Parámetros de recompensas
rewards = {
    "reward_speed_weight": 0.1,             # Peso de la velocidad en la recompensa
    "reward_track_position_weight": 5,      # Peso por alcanzar un checkpoint 
    "reward_laps_weight": 500.0,            # Peso de las vueltas completadas DEBE SER ALTO
    "penalty_low_rpms": -0.2,               # Penalización por quedarse quieto
    "penalty_backwards": -0.5,              # Penalización por ir hacia atrás
    "penalty_tyres_out": -0.5,              # Penalización por salirse de la pista
    "penalty_car_damage": -2.0,
    "threshold_speed": 10.0,                # Velocidad mínima para recibir recompensa por velocidad
    "threshold_rpms": 2000.0,               # RPMs mínimas para no recibir recompensa negativa por quedarse quieto
    "threshold_checkpoint": 0.01            # Umbral de posición en la pista para recibir recompensa por posición
    } 

# Hiperparametros de SAC
learning_rate = 3e-4   # Tasa de aprendizaje para el optimizador
discount_factor = 0.99 # Factor de descuento para las recompensas futurasp
alpha = 0.2            # Parámetro de entropía para SAC (controla la exploración)
tau = 0.005            # Parámetro de actualización suave de las redes objetivo
batch_size = 120     # Tamaño de batch para actualizar el agente

# Otras configuraciones
max_steps_per_episode = 20   # Máximo número de pasos por episodio
num_episodes = 500              # Número de episodios de entrenamiento
save_interval = 50 

# Crear un callback para guardar el modelo en intervalos regulares
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=save_dir, name_prefix="SAC_model")

# Definir las transformaciones de las imagenes
transform = transforms.Compose([
    transforms.Resize((input_size)),    # Cambia el tamaño de las imágenes a (height x width)
    transforms.ToTensor(),              # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# Definir el entorno personalizado
class RacingEnv(gym.Env):
    def __init__(self, model):
        super(RacingEnv, self).__init__()
        self.controller = XboxControllerEmulator()  
        self.region = get_region(screen_size, True)
        self.previous_variables = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Definir espacios de acción y observación
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-2.1179,  # Aproximado de (0 - 0.485) / 0.229 para el canal rojo
            high=2.2489,  # Aproximado de (1 - 0.406) / 0.224 para el canal azul
            shape=(3, *input_size), 
            dtype=np.float32
        )

        self.current_image = None  # Para almacenar la imagen capturada

        # Iniciar hilo de captura
        self.lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.capture_thread.start()      

    def capture_images(self):
        while not stop_event.is_set():
            start_time = time.time()
            with self.lock:
                self.current_image = capture_and_process(self.region, transform, self.device)  # Actualiza la imagen
                time.sleep(max(0, 1/fps - (time.time() - start_time)))    

    def get_observation(self):
        return self.current_image  # Retorna la última imagen capturada
        
    def apply_action(self, action):
        self.controller.steering(action[0])  # Enviar la acción de dirección al simulador
        self.controller.throttle_break(action[1])  # Enviar la acción de aceleración/freno al simulador

    def reset(self):
        # Asegurar que el hilo de captura está activo y que la imagen no es None
        if not self.capture_thread.is_alive():
            raise RuntimeError("El hilo de captura no está activo.")
        while self.current_image is None:
            time.sleep(0.01)  # Espera a que haya una imagen capturada
        
        self.controller.reset()

        # reset_environment()  # Reiniciar el simulador si es necesario

        obs = self.get_observation()
        self.previous_variables = {
            "previous_checkpoint": 0.0,
            "previous_position": 0.0,
            "previous_lap": 0
        }
        return obs

    def step(self, action):
        if not self.capture_thread.is_alive():
            raise RuntimeError("El hilo de captura no está activo.")
        if self.current_image is None:
            raise RuntimeError("No hay imagen disponible para el paso actual.")
        
        with self.lock:
            # Enviar la acción al entorno
            self.apply_action(action)
            obs = self.get_observation()
            #variables = udp_listener()
            variables = { # Placeholder para las variables del entorno
                    "speed": 0.0,
                    "rpms": 0,
                    "laps": 0,
                    "track_position": 0.0,
                    "tyres_out": 0,
                    "car_damage": 0.0,
                    "transmitting": False
                    }
            reward, done = calculate_reward(variables, rewards, self.previous_variables)
            info = {}
            return obs, reward, done, info

    def predict_action(self):
        obs = self.get_observation()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # La CNN toma la observación procesada y genera una acción
                action_values = self.model(obs)
                # Limita los valores de acción y los convierte en una lista para usarlos en el entorno
                action = torch.clamp(action_values, min=-1.0, max=1.0).tolist()[0]
        return action

    def close(self):
        self.controller.reset()
        stop_event.set()        
        key_thread.join()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.capture_thread.join()

# Inicializar y verificar el entorno
env = RacingEnv()
check_env(env)

# Entrenar el modelo SAC
model = SAC("CnnPolicy", env, verbose=1, learning_rate=3e-4, batch_size=120, tau=0.005, gamma=0.99, ent_coef=0.2, train_freq=1, target_update_interval=1, device="auto")
model.learn(total_timesteps=50000, callback=checkpoint_callback)

# Guardar el modelo
model.save(f"{save_dir}/final_SAC_model")
print("Modelo entrenado y guardado.")