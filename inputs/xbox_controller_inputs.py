import pygame
import time
import sys
import logging
from typing import Tuple

class XboxControllerReader:
    """
    Lee el estado actual de un control de Xbox.
    También puede funcionar con otros controles similares.
    """

    joystick: pygame.joystick
    name: str
    joystick_id: int

    def __init__(self, total_wait_secs: int = 10):
        """
        Inicializa el controlador.
        
        - total_wait_secs: Número de segundos a esperar antes de comenzar a leer el estado del controlador.
          Pygame tarda un tiempo en inicializarse, durante los primeros segundos se pueden obtener lecturas incorrectas.
          Se recomienda esperar algunos segundos antes de empezar a leer los datos.
        """
        pygame.init()
        pygame.joystick.init()
        try:
            # Intenta inicializar el controlador conectando el primer control
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except pygame.error:
            logging.warning(
                "No se encontró un controlador. Asegúrate de que el controlador esté conectado y sea reconocido por Windows."
            )
            sys.exit()

        # Obtiene el nombre y el ID del controlador
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

    def read(self) -> Tuple[str, str]:
        """
        Lee el estado actual del controlador.

        Salida:
        - lx: Valor actual del eje X del stick izquierdo, en el rango [-1, 1]
        - lt: Valor actual del gatillo izquierdo, en el rango [-1, 1]
        - rt: Valor actual del gatillo derecho, en el rango [-1, 1]
        """
        #_ = pygame.event.get()
        pygame.event.pump()  # Actualiza el estado de los eventos del joystick
        lx, lt, rt = (
            self.joystick.get_axis(0),
            self.joystick.get_axis(4),
            self.joystick.get_axis(5),
        )

        steering = f"{lx:.2f}"
        throttle_brake = f"{(rt - lt) / 2:.2f}"

        return steering, throttle_brake

        #return lx, lt, rt
    

def imprimir_estado_controlador() -> None:
    """
    Función de prueba que imprime los valores del joystick en la terminal.
    """
    control = XboxControllerReader()

    print("Leyendo el estado del controlador de Xbox (Ctrl+C para salir)...\n")

    try:
        while True:
            steering, throttle_brake = control.read()  # Llama al método read
            #print(f"Valor del stick izquierdo (lx): {lx:.2f}, Gatillo izquierdo (lt): {lt:.2f}, Gatillo derecho (rt): {rt:.2f}", end="\r")
            print(f"Dirección: {steering}, Acelerar frenar: {throttle_brake}", end="\r")
            time.sleep(0.1)  # Pausa corta para evitar un loop muy rápido
    except KeyboardInterrupt:
        print("\nSaliendo...")


# Ejecutar la función para leer el estado del controlador
#imprimir_estado_controlador()