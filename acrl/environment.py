
# local imports
from rewards import RewardFunction
from UDP_listener import udp_listener
from inputs.xbox_controller_emulator import XboxControllerEmulator
from inputs.GameInputs import reset_race
from window_interface import MSSWindowInterface

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

from rtgym import RealTimeGymInterface
import tmrl.config.config_constants as cfg


class AC_Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Assetto Corsa.
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False, #POR IMPLEMENTAR
                 grayscale: bool = cfg.GRAYSCALE,
                 resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                 human_mode: bool = cfg.TMRL_CONFIG["HUMAN_WORKER"]
                 ):
        """
        Base rtgym interface for Assetto Corsa 

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.controller = None
        self.window_interface = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to if resize_to != (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT) else None
        print(f"\nResizing images to {self.resize_to}\n")
        self.human_mode = human_mode
        self.fullscreen = cfg.ENV_CONFIG['FULL_SCREEN']
        self.initialized = False
        self.ep_rew = []
        self.action = None
        self.best = 0.0

        # Crear el visualizador
        self.visualizer = ImageVisualizer()

    def initialize_common(self):    

        if self.gamepad:
            if self.human_mode:
                logging.debug(" real joystick in use")
            else:
                self.controller = XboxControllerEmulator()
                logging.debug(" virtual joystick in use")
        
        assert self.gamepad == True, "Only gamepad supported"


        while True:
            try:
                self.window_interface = MSSWindowInterface("Assetto Corsa",)
                break
            except Exception as e:
                print("Waiting for Assetto Corsa's window...                                                                        ", end="\r")
                time.sleep(0.1)

        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        
        """ if not self.fullscreen:
            self.window_interface.move_and_resize() """

        self.reward_function = RewardFunction(max_mistakes=cfg.REWARD_CONFIG['MAX_MISTAKES'],
                                                steps_to_forget=cfg.REWARD_CONFIG['STEPS_TO_FORGET'],
                                                min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                                reward_checkpoint=cfg.REWARD_CONFIG['REWARD_CHECKPOINT'],
                                                reward_progress=cfg.REWARD_CONFIG['REWARD_PROGRESS'],
                                                reward_laps_weight=cfg.REWARD_CONFIG['LAPS_WEIGHT'],
                                                start_up_multiplier=cfg.REWARD_CONFIG['START_UP_MULTIPLIER'],
                                                penalty_no_progress=cfg.REWARD_CONFIG['PENALTY_NO_PROGRESS'],
                                                penalty_low_speed=cfg.REWARD_CONFIG['PENALTY_LOW_SPEED'],
                                                penalty_backwards=cfg.REWARD_CONFIG['PENALTY_BACKWARDS'],
                                                penalty_tyres_out=cfg.REWARD_CONFIG['PENALTY_TYRES_OUT'],
                                                penalty_car_damage=cfg.REWARD_CONFIG['PENALTY_CAR_DAMAGE'],
                                                penalty_collision=cfg.REWARD_CONFIG['PENALTY_COLLISION'],
                                                penalty_non_smooth_actions=cfg.REWARD_CONFIG['PENALTY_NON_SMOOTH_ACTIONS'],
                                                threshold_speed=cfg.REWARD_CONFIG['THRESHOLD_SPEED'],
                                                threshold_rpms=cfg.REWARD_CONFIG['THRESHOLD_RPMS'],
                                                threshold_checkpoint=cfg.REWARD_CONFIG['THRESHOLD_CHECKPOINT'],
                                                threshold_smooth_actions=cfg.REWARD_CONFIG['THRESHOLD_SMOOTH_ACTIONS'],
                                                threshold_damage=cfg.REWARD_CONFIG['THRESHOLD_DAMAGE'],
                                                hist_len=self.img_hist_len,
                                                time_step_duration=cfg.ENV_CONFIG['RTGYM_CONFIG']['time_step_duration']
                                                )
                                              

    def initialize(self):
        self.initialize_common()
        while not self.grab_data()["transmitting"]:
            time.sleep(0.1)
            print("Waiting for telemetry data...                                                                             ", end="\r")
        print("Telemetry data received\n")
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [gas-brake,steering] values between -1.0 and 1.0
        """
        self.action = control

        if self.gamepad:
            if control is not None:
                if not self.human_mode:
                    self.controller.control_gamepad(control)          
        else:
            pass
            # Por implementar para AC
            """ if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions) """

    def grab_data(self):
        """
        Non-blocking function
        Grabs the telemetry data from the UDP socket
        """
        data = udp_listener()
        return data

    def grab_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering        

        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # shape is (height, width) for cv2 grayscale images
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #img[:, :, ::-1]  # reversed view for numpy RGB convention
            # shape is (height, width, channels) for cv2 images    
        self.visualizer.update_image(img)  # Actualizar la imagen en el visualizador
        return img

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        if not self.human_mode:
            self.send_control(self.get_default_action())
            reset_race(cfg.SLEEP_TIME_AT_RESET)
            time.sleep(0.5)
            self.controller.next_gear()
        else:
            reset_race(cfg.SLEEP_TIME_AT_RESET)
        # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        
        if len(self.ep_rew) > 0:
            # Calcula estadísticas del episodio anterior
            total_reward = sum(self.ep_rew)
            min_reward = min(self.ep_rew)
            max_reward = max(self.ep_rew)
            avg_reward = total_reward / len(self.ep_rew)
            print(f"PR: {self.best} Episode reward: {total_reward:.2f} Min reward: {min_reward:.2f} Max reward: {max_reward:.2f} Average reward: {avg_reward:.2f} \n")
        else:
            print("No rewards recorded for the previous episode.\n")
        self.ep_rew = []
        
        self.reset_common()
        data = self.grab_data()
        img = self.grab_img()

        speed = np.array([
            data["speed"],
        ], dtype='float32')
        gear = np.array([
            data["gear"],
        ], dtype='float32')
        rpm = np.array([
            data["rpms"],
        ], dtype='float32')

        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}


    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        if not self.human_mode:
            self.send_control(self.get_default_action())
        if self.save_replays:
            # TODO: POR IMPLEMENTAR
            pass

        """ self.reset_race()
        time.sleep(0.5) """
        

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data = self.grab_data()
        img = self.grab_img()

        speed = np.array([
            data["speed"],
        ], dtype='float32')
        gear = np.array([
            data["gear"],
        ], dtype='float32')
        rpm = np.array([
            data["rpms"],
        ], dtype='float32')

        rew, terminated = self.reward_function.compute_reward(data, self.action)
        self.ep_rew.append(rew)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]        
        info = {}       
        rew = np.float32(rew)

        if data["track_position"] > self.best and data["track_position"] < 0.995:
            self.best = data["track_position"]

        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=10, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))

        # en caso de normalizar los colores, no se muy bien de donde viene esto:
        """ spaces.Box(
            low=-2.1179,  # Aproximado de (0 - 0.485) / 0.229 para el canal rojo
            high=2.6400,  # Aproximado de (1 - 0.406) / 0.224 para el canal azul
            shape=self.input_size, 
            dtype=np.float32
        ) """

        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len * 3, h, w))#, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(2, )) # Cambiar a (3,) para gas brake y steering

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0], dtype='float32') # Cambiar a [0.0, 0.0, 0.0] para gas brake y steering

import tkinter as tk  
from PIL import Image, ImageTk

class ImageVisualizer:
    def __init__(self, title="Visualización en tiempo real", position="top-right"):
        """
        Inicializa el visualizador de imágenes con Tkinter.

        Args:
            title (str): Título de la ventana.
            position (str): Posición de la ventana en la pantalla ("top-right", "top-left", etc.).
        """
        self.root = tk.Tk()
        self.root.title(title)

        # Configurar la posición de la ventana
        self.set_window_position(position)

        # Crear un widget de etiqueta para mostrar la imagen
        self.label = tk.Label(self.root)
        self.label.pack()

    def set_window_position(self, position):
        """
        Configura la posición de la ventana en la pantalla.

        Args:
            position (str): Posición de la ventana ("top-right", "top-left", etc.).
        """
        # Obtener el tamaño de la pantalla
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calcular la posición
        if position == "top-right":
            x_offset = screen_width - 300  # Dejar un margen pequeño
            y_offset = 0
        elif position == "top-left":
            x_offset = 0
            y_offset = 0
        elif position == "bottom-right":
            x_offset = screen_width - 300
            y_offset = screen_height - 300
        elif position == "bottom-left":
            x_offset = 0
            y_offset = screen_height - 300
        else:  # Centro por defecto
            x_offset = screen_width // 2
            y_offset = screen_height // 2

        # Configurar la geometría de la ventana (sin ajustar dimensiones manualmente)
        self.root.geometry(f"+{x_offset}+{y_offset}")

    def update_image(self, img):
        """
        Actualiza la imagen mostrada en la ventana.

        Args:
            img: La imagen a mostrar (numpy array).
        """
        # Convertir la imagen a un formato compatible con Tkinter
        if len(img.shape) == 2:  # Escala de grises
            img = Image.fromarray(img)
        else:  # Color RGB
            img = Image.fromarray(img, 'RGB')

        # Convertir la imagen a un objeto PhotoImage
        img_tk = ImageTk.PhotoImage(img)

        # Actualizar la imagen en el widget de etiqueta
        self.label.config(image=img_tk)
        self.label.image = img_tk

        # Actualizar la ventana
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        """
        Cierra la ventana de visualización.
        """
        self.root.destroy()
    
if __name__ == "__main__":
    pass