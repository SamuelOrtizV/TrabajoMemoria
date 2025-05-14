
# local imports
from window_interface import MSSWindowInterface

# standard library imports
import logging
import time
from collections import deque
from threading import Thread

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone
import tmrl.config.config_constants as cfg


class DummyRCDroneInterface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control the dummydrone.
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 grayscale: bool = cfg.GRAYSCALE,
                 resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
                 ):
        
        self.rc_drone = None
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.initialized = False
        self.blank_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        self.rendering_thread = Thread(target=self._rendering_thread, args=(), kwargs={}, daemon=True)

        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.reward_function = None
        self.window_interface = None
        
        self.resize_to = resize_to
        self.grayscale = grayscale
        self.initialized = False
        self.ep_rew = []
        self.action = None
        self.best = 0.0
        self.time_step_duration = cfg.ENV_CONFIG['RTGYM_CONFIG']['time_step_duration']

        # Crear el visualizador
        self.visualizer = ImageVisualizer()

    def _rendering_thread(self):
        from time import sleep
        while True:
            sleep(0.1)
            self.render()

    def initialize_common(self):    
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)

        self.rc_drone = DummyRCDrone()
        self.rendering_thread.start()
        
        while True:
            try:
                self.window_interface = MSSWindowInterface("Dummy RC drone", False)
                break
            except Exception as e:
                print("Waiting for window...                                                                        ", end="\r")
                time.sleep(0.1)  
                                              
    def initialize(self):
        self.initialize_common()
        self.initialized = True

    def send_control(self, control):
        vel_x = control[0]
        vel_y = control[1]
        self.rc_drone.send_control(vel_x, vel_y)

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
        self.target[0] = np.random.uniform(-0.5, 0.5)
        self.target[1] = np.random.uniform(-0.5, 0.5)
        

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
        img = self.grab_img()

        pos_x, pos_y = self.rc_drone.get_observation()

        speed = np.array([
            0,
        ], dtype='float32')
        gear = np.array([
            0,
        ], dtype='float32')
        rpm = np.array([
            0,
        ], dtype='float32')

        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]

        return obs, {}


    def wait(self):
        pass
        

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """

        """while not self.grab_data()["transmitting"]:
            time.sleep(0.1)
            print("Waiting for telemetry data...                                                                             ", end="\r") """

        pos_x, pos_y = self.rc_drone.get_observation()
        img = self.grab_img()

        speed = np.array([
            0,
        ], dtype='float32')
        gear = np.array([
            0,
        ], dtype='float32')
        rpm = np.array([
            0,
        ], dtype='float32')

        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        terminated = rew > -0.01

        self.ep_rew.append(rew)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]        
        info = {}       
        rew = np.float32(rew)

        return obs, rew, terminated, info
    
    def render(self):
        image = self.blank_image.copy()
        pos_x, pos_y = self.rc_drone.get_observation()
        image = cv2.circle(img=image,
                           center=(int(pos_x * 200) + 250, int(pos_y * 200) + 250),
                           radius=10,
                           color=(255, 0, 0),
                           thickness=2)
        image = cv2.circle(img=image,
                           center=(int(self.target[0] * 200) + 250, int(self.target[1] * 200) + 250),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
        cv2.imshow("Dummy RC drone", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=-1, high=1, shape=(1, ))
        gear = spaces.Box(low=-1, high=1, shape=(1, ))
        rpm = spaces.Box(low=-1, high=1, shape=(1, ))

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
        return spaces.Box(low=-2.0, high=2.0, shape=(2,))

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

# rtgym configuration dictionary:

DUMMY_RC_DRONE_CONFIG = DEFAULT_CONFIG_DICT.copy()
DUMMY_RC_DRONE_CONFIG["interface"] = DummyRCDroneInterface
DUMMY_RC_DRONE_CONFIG["time_step_duration"] = 0.05
DUMMY_RC_DRONE_CONFIG["start_obs_capture"] = 0.05
DUMMY_RC_DRONE_CONFIG["time_step_timeout_factor"] = 1.0
DUMMY_RC_DRONE_CONFIG["ep_max_length"] = 100
DUMMY_RC_DRONE_CONFIG["act_buf_len"] = 2
DUMMY_RC_DRONE_CONFIG["reset_act_buf"] = False
DUMMY_RC_DRONE_CONFIG["benchmark"] = True
DUMMY_RC_DRONE_CONFIG["benchmark_polyak"] = 0.2


if __name__ == "__main__":
    pass