
# local imports
from rewards import RewardFunction
from UDP_listener import udp_listener
from inputs.xbox_controller_emulator import XboxControllerEmulator
from inputs.GameInputs import reset_race

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.window import WindowInterface


class AC_Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Assetto Corsa.
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False, #POR IMPLEMENTAR
                 grayscale: bool = cfg.GRAYSCALE,
                 resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT)):
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
        self.fullscreen = cfg.ENV_CONFIG['FULL_SCREEN']
        self.initialized = False
        self.best = 0.0

    def initialize_common(self):
        if self.gamepad:
            self.controller = XboxControllerEmulator()
            logging.debug(" virtual joystick in use")
        while True:
            try:
                self.window_interface = WindowInterface("Assetto Corsa",)
                break
            except Exception as e:
                print("Waiting for Assetto Corsa's window...                                                                        ", end="\r")
                time.sleep(0.1)

        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        
        if not self.fullscreen:
            self.window_interface.move_and_resize()

        self.reward_function = RewardFunction(max_mistakes=cfg.REWARD_CONFIG['MAX_MISTAKES'],
                                                steps_to_forget=cfg.REWARD_CONFIG['STEPS_TO_FORGET'],
                                                min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                                reward_speed_weight=cfg.REWARD_CONFIG['SPEED_WEIGHT'],
                                                reward_track_position_weight=cfg.REWARD_CONFIG['TRACK_POS_WEIGHT'],
                                                reward_laps_weight=cfg.REWARD_CONFIG['LAPS_WEIGHT'],
                                                penalty_low_rpms=cfg.REWARD_CONFIG['PENALTY_LOW_RPMS'],
                                                penalty_backwards=cfg.REWARD_CONFIG['PENALTY_BACKWARDS'],
                                                penalty_tyres_out=cfg.REWARD_CONFIG['PENALTY_TYRES_OUT'],
                                                penalty_car_damage=cfg.REWARD_CONFIG['PENALTY_CAR_DAMAGE'],
                                                threshold_speed=cfg.REWARD_CONFIG['THRESHOLD_SPEED'],
                                                threshold_rpms=cfg.REWARD_CONFIG['THRESHOLD_RPMS'],
                                                threshold_checkpoint=cfg.REWARD_CONFIG['THRESHOLD_CHECKPOINT']
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
            control: np.array: [gas,brake,steering] values between -1.0 and 1.0
        """
        if self.gamepad:
            if control is not None:
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
            
        print(f"Best lap: {self.best}  Gas Brake Turn: {np.round(control, 2)} ", end="\r")

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
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
            # shape is (height, width, channels) for cv2 images
        # img = img.astype(np.float32) / 255.0
        return img

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        reset_race(cfg.SLEEP_TIME_AT_RESET)
        # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data = self.grab_data()
        img = self.grab_img()

        # Estos datos son un poco trampa ya que no se extraen de forma visual
        # Hay que evaluar su impacto y la posibilidad de extraerlos de forma visual
        # o incluso de no usarlos
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
        self.send_control(self.get_default_action())
        if self.save_replays:
            # POR IMPLEMENTAR
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

        rew, terminated = self.reward_function.compute_reward(data)
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
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')
    
# rtgym configuration ductionary:

AC_ENV_CONFIG = DEFAULT_CONFIG_DICT.copy()
AC_ENV_CONFIG["interface"] = AC_Interface


if __name__ == "__main__":
    pass