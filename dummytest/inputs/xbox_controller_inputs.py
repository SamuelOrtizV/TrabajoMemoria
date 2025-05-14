import pygame
import time
import sys
import logging
import numpy as np
from typing import Tuple

class XboxControllerReader:
    """
    Lee el estado actual de un control de Xbox.
    También puede funcionar con otros controles similares.
    """

    joystick: pygame.joystick
    name: str
    joystick_id: int

    def __init__(self, total_wait_secs: int = 5):
        """
        Inicializa el controlador.
        
        - total_wait_secs: Número de segundos a esperar antes de comenzar a leer el estado del controlador.
          Pygame tarda un tiempo en inicializarse, durante los primeros segundos se pueden obtener lecturas incorrectas.
          Se recomienda esperar algunos segundos antes de empezar a leer los datos.
        """
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.name = None
        self.joystick_id = None

        self._initialize_controller(total_wait_secs)

    def _initialize_controller(self, total_wait_secs: int):
        """
        Intenta inicializar el controlador.
        """
        try:
            if pygame.joystick.get_count() == 0:
                raise pygame.error("No se encontró un controlador conectado.")

            # Intenta inicializar el primer controlador disponible
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.name = self.joystick.get_name()
            self.joystick_id = self.joystick.get_id()

            # Espera algunos segundos antes de comenzar a leer el controlador
            for delay in range(int(total_wait_secs), 0, -1):
                print(
                    f"Inicializando la lectura del controlador, esperando {delay} segundos para evitar lecturas incorrectas...",
                    end="\r",
                )
                time.sleep(1)

            print(f"Capturando entrada de: {self.name} (ID: {self.joystick_id})\n")

        except pygame.error as e:
            logging.warning(f"No se encontró un controlador: {e}")
            self.joystick = None

    def _check_connection(self):
        """
        Verifica si el controlador sigue conectado.
        Si no está conectado, intenta reconectarlo.
        """
        if self.joystick is None or not self.joystick.get_init():
            print("Controlador desconectado. Intentando reconectar...")
            self._initialize_controller(total_wait_secs=1)

    def read(self) -> np.ndarray:
        """
        Lee el estado actual del controlador.

        Salida:
        - np.ndarray: Array con [throttle_brake, steering]
        - bool: Indica si el controlador está conectado
        """
        self._check_connection()

        if self.joystick is None:
            # Si no hay controlador conectado, devuelve valores por defecto
            return np.array([0.0, 0.0], dtype=np.float32)

        try:
            pygame.event.pump()  # Actualiza el estado de los eventos del joystick
            lx, lt, rt = (
                self.joystick.get_axis(0),
                self.joystick.get_axis(4),
                self.joystick.get_axis(5),
            )

            steering = lx
            throttle_brake = (rt - lt) / 2

            return np.array([throttle_brake, steering], dtype=np.float32)
        except pygame.error as e:
            logging.warning(f"Error al leer el controlador: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)


def imprimir_estado_controlador() -> None:
    """
    Función de prueba que imprime los valores del joystick en la terminal.
    """
    control = XboxControllerReader()

    print("Leyendo el estado del controlador de Xbox (Ctrl+C para salir)...\n")

    try:
        while True:
            entrada, conectado = control.read()  # Llama al método read
            if conectado:
                print(f"Dirección: {entrada[1]:.2f}, Acelerar/Frenar: {entrada[0]:.2f}", end="\r")
            else:
                print("Esperando reconexión del controlador...", end="\r")
            time.sleep(0.1)  # Pausa corta para evitar un loop muy rápido
    except KeyboardInterrupt:
        print("\nSaliendo...")


# Ejecutar la función para leer el estado del controlador

if __name__ == "__main__":
    imprimir_estado_controlador()