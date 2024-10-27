import vgamepad as vg
import time

class XboxControllerEmulator:
    def __init__(self):
        # Crear una instancia de un gamepad de Xbox virtual
        self.gamepad = vg.VX360Gamepad()
        self.wait_before_start = 5  # Segundos antes de comenzar a leer el controlador

        # Espera algunos segundos antes de comenzar a emular el controlador
        for delay in range(int(self.wait_before_start), 0, -1):
            print(
                f"Inicializando emulación del controlador, esperando {delay} segundos para evitar lecturas incorrectas...",
                end="\r",
            )
            time.sleep(1)

    def steering(self, x_value: float):
        """
        Mueve el joystick izquierdo horizontalmente.
        x_value: Valor entre -1.0 y 1.0, donde -1.0 es izquierda y 1.0 es derecha.
        """
        # Limitar el valor entre -1.0 y 1.0
        x_value = max(min(x_value, 1.0), -1.0)
        
        # Mover el joystick horizontalmente (x), mantener vertical (y) en el centro (0.0)
        self.gamepad.left_joystick_float(x_value, 0.0)
        self.gamepad.update()  # Actualizar el estado del gamepad

    def throttle_break(self, value: float):
        """
        Presiona uno de los gatillos.
        value: Valor entre -1.0 (freno al máximo) 0.0 (sin presión) y 1.0 (aceleración máxima).
        """
        # Limitar el valor entre -1.0 y 1.0
        value = max(min(value, 1.0), -1.0)
        
        if value < 0:
            self.gamepad.left_trigger_float(-value)  # Gatillo izquierdo
        else:
            self.gamepad.right_trigger_float(value)  # Gatillo derecho

        self.gamepad.update()  # Actualizar el estado del gamepad


    def reset(self):
        """
        Resetea todos los controles del gamepad.
        """
        self.gamepad.reset()
        self.gamepad.update()  # Actualizar el estado del gamepad

""" # Crear una instancia del emulador de control
controller = XboxControllerEmulator()

print("Empezando a enviar comandos al controlador...")

time.sleep(5)

input("Presiona Enter para empezar...")

while True:
    # Presionar el gatillo derecho al máximo
    controller.throttle_break(1.0)
    print("Gatillo derecho presionado al máximo")
    time.sleep(3)
    
    # Mover el joystick izquierdo hacia la derecha (x = 1.0)
    controller.steering(1.0)
    print("Joystick izquierdo hacia la derecha")
    time.sleep(2)

    # Mover el joystick izquierdo hacia la izquierda (x = -1.0)
    controller.steering(-1.0)
    print("Joystick izquierdo hacia la izquierda")
    time.sleep(2)

    # Resetear todo
    controller.reset()

    # Presionar gatillo izquierdo al máximo
    controller.throttle_break(-1.0)
    print("Gatillo izquierdo presionado al máximo")
    time.sleep(3)

    # Resetear todo
    controller.reset() """

