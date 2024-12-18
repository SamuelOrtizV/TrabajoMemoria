# Definir el entorno personalizado
import gym
from gym import spaces
import torch
import numpy as np
import threading
import time
from inputs.xbox_controller_emulator import XboxControllerEmulator
from inputs.GameInputs import reset_environment
from UDP_listener import udp_listener
from model import get_region
from ScreenRecorder import capture_screen
from PIL import Image

def calculate_reward(variables, rewards, previous_location):
    """Calcula la recompensa basándose en las variables del entorno
    Args:
        variables (dict): Diccionario con las variables del entorno
        rewards (dict): Diccionario con los pesos de las recompensas y penalizaciones
        previous_location (dict): Diccionario con la ubicación anterior del auto
    Returns:
        float: Recompensa calculada"""

    reward = 0.0

    # Recompensas
    if variables["laps"] > previous_location["previous_lap"]:  # Si se ha completado una vuelta
        reward += rewards["reward_laps_weight"]
        previous_location.update({"previous_checkpoint": 0.0, "previous_position": 0.0, "previous_lap": variables["laps"]})

    track_position = 0.0 if variables["track_position"] == 1.0 else variables["track_position"]

    position_difference = track_position - previous_location["previous_position"]
    checkpoint_difference = track_position - previous_location["previous_checkpoint"]

    reward += rewards["reward_speed_weight"] if variables["speed"] >= rewards["threshold_speed"] else -rewards["reward_speed_weight"]

    if checkpoint_difference > rewards["threshold_checkpoint"]:  # Si se ha avanzado en la pista
        print(f"Se ha alcanzado un checkpoint: {track_position}")
        reward += rewards["reward_track_position_weight"]
        previous_location["previous_checkpoint"] = track_position

    # Penalizaciones
    reward += rewards["penalty_tyres_out"] * variables["tyres_out"] if variables["tyres_out"] > 0 else 0  # Penalizar por ruedas salidas de la pista
    reward += rewards["penalty_car_damage"] if variables["car_damage"] > 0 else 0                         # Penalizar por daño al auto
    reward += rewards["penalty_low_rpms"] if variables["rpms"] < rewards["threshold_rpms"] else 0         # Penalizar por quedarse quieto
    reward += rewards["penalty_backwards"] if position_difference < 0 else 0                              # Penalizar por ir hacia atrás

    done = variables["tyres_out"] == 4 or variables["car_damage"] > 0  # El episodio termina si el auto está fuera de la pista o dañado

    return reward, done

class AC_Env(gym.Env):
    def __init__(self, model, screen_size, input_size, transform, rewards, fps, stop_event):

        super(AC_Env, self).__init__()
        self.controller = XboxControllerEmulator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.screen_size = screen_size
        self.input_size = input_size
        self.transform = transform
        self.rewards = rewards
        self.fps = fps
        self.stop_event = stop_event
        self.region = get_region(screen_size, True)
        self.previous_variables = None
        

        # Definir espacios de acción y observación
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-2.1179,  # Aproximado de (0 - 0.485) / 0.229 para el canal rojo
            high=2.6400,  # Aproximado de (1 - 0.406) / 0.224 para el canal azul
            shape=(3, *input_size), 
            dtype=np.float32
        )

        self.current_image = None  # Para almacenar la imagen capturada

        # Iniciar hilo de captura
        self.lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.capture_thread.start()      

    def capture_images(self):
        while not self.stop_event.is_set():
            start_time = time.time()
            with self.lock:
                img = capture_screen(self.region)
                img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                self.current_image = self.transform(img).unsqueeze(0).to(self.device)
                # Actualiza la imagen
                time.sleep(max(0, 1/self.fps - (time.time() - start_time)))   

    def get_observation(self):
        return self.current_image  # Retorna la última imagen capturada
        
    def apply_action(self, action):
        self.controller.steering(action[0])  # Enviar la acción de dirección al simulador
        self.controller.throttle_break(action[1])  # Enviar la acción de aceleración/freno al simulador

    def reset(self):
        self.done = False
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
            reward, done = calculate_reward(variables, self.rewards, self.previous_variables)
            info = {}
            return obs, reward, done, info

    def predict_action(self):
        obs = self.get_observation()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # La CNN toma la observación procesada y genera una acción
                action_values = self.model(obs)
                # Limita los valores de acción y los convierte en una lista para usarlos en el entorno
                #action = torch.clamp(action_values, min=-1.0, max=1.0).tolist()[0]
                action = torch.clamp(action_values, min=-1.0, max=1.0).cpu().numpy()[0] # Hay que usar detach() para evitar errores?
        return action

    def close(self):
        self.controller.reset()
        self.stop_event.set()        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.capture_thread.join()