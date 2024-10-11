from xbox_controller_emulator import XboxControllerEmulator
import time

# Crear una instancia del emulador de control
controller = XboxControllerEmulator()

print("Empezando a enviar comandos al controlador...")

time.sleep(5)

while True:
      # Mover el joystick izquierdo hacia la derecha (x = 1.0)
      controller.steering(1.0)
      time.sleep(1)

      # Mover el joystick izquierdo hacia la izquierda (x = -1.0)
      controller.steering(-1.0)
      time.sleep(1)

      # Presionar el gatillo derecho al máximo
      controller.throttle_break(1.0)
      time.sleep(1)

      # Soltar el gatillo derecho
      controller.throttle_break(-1.0)

      # Resetear todo
      controller.reset()
