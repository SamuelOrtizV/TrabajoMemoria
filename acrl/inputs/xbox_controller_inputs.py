import pygame
import time
import sys
import logging
import numpy as np
from typing import Tuple

class XboxControllerReader:
    """Read the current state of an Xbox controller (or compatible).

    Returns normalized values in [-1.0, 1.0] for:
    - throttle_brake (>=0 throttle, <0 brake)
    - steering (left stick X)
    """

    joystick: pygame.joystick
    name: str
    joystick_id: int

    def __init__(self, total_wait_secs: int = 5):
        """Initialize pygame and the first available controller.

        total_wait_secs: seconds to wait before reading to avoid early misreads
        while pygame initializes the device.
        """
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.name = None
        self.joystick_id = None

        self._initialize_controller(total_wait_secs)

    def _initialize_controller(self, total_wait_secs: int):
        """Attempt to initialize the controller; wait a bit for stability."""
        try:
            if pygame.joystick.get_count() == 0:
                raise pygame.error("No controller detected.")

            # Intenta inicializar el primer controlador disponible
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.name = self.joystick.get_name()
            self.joystick_id = self.joystick.get_id()

            # Wait a few seconds before starting to read the controller
            for delay in range(int(total_wait_secs), 0, -1):
                print(f"Initializing controller read in {delay} s...", end="\r")
                time.sleep(1)

            print(f"Reading input from: {self.name} (ID: {self.joystick_id})\n")

        except pygame.error as e:
            logging.warning(f"No controller found: {e}")
            self.joystick = None

    def _check_connection(self):
        """Verify controller is still connected; try to reconnect otherwise."""
        if self.joystick is None or not self.joystick.get_init():
            print("Controller disconnected. Trying to reconnect...")
            self._initialize_controller(total_wait_secs=1)

    def read(self) -> np.ndarray:
        """Read the current controller state.

        Returns a numpy array [throttle_brake, steering].
        """
        self._check_connection()

        if self.joystick is None:
            # If no controller is connected, return default neutral values
            return np.array([0.0, 0.0], dtype=np.float32)

        try:
            pygame.event.pump()  # Update joystick event state
            lx, lt, rt = (
                self.joystick.get_axis(0),
                self.joystick.get_axis(4),
                self.joystick.get_axis(5),
            )

            steering = lx
            throttle_brake = (rt - lt) / 2

            return np.array([throttle_brake, steering], dtype=np.float32)
        except pygame.error as e:
            logging.warning(f"Error reading controller: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)


def print_controller_state() -> None:
    """Demo: print joystick values in the terminal until Ctrl+C."""
    control = XboxControllerReader()

    print("Reading Xbox controller state (Ctrl+C to quit)...\n")

    try:
        while True:
            vals = control.read()
            connected = control.joystick is not None and control.joystick.get_init()
            if connected:
                print(f"Steering: {vals[1]:.2f}, Throttle/Brake: {vals[0]:.2f}", end="\r")
            else:
                print("Waiting for controller reconnection...", end="\r")
            time.sleep(0.1)  # small pause to avoid a too-fast loop
    except KeyboardInterrupt:
        print("\nExiting...")


# Ejecutar la función para leer el estado del controlador

if __name__ == "__main__":
    print_controller_state()