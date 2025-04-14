import vgamepad as vg
import time

class XboxControllerEmulator:
    def __init__(self):
        # Crear una instancia de un gamepad de Xbox virtual
        self.gamepad = vg.VX360Gamepad()
        self.wait_before_start = 1 #5  # Segundos antes de comenzar a leer el controlador

        # Espera algunos segundos antes de comenzar a emular el controlador
        for delay in range(int(self.wait_before_start), 0, -1):
            print(
                f"Inicializando emulación del controlador, esperando {delay} segundos para evitar lecturas incorrectas...",
                end="\r",
            )
            time.sleep(1)

    def control_gamepad(self, control):
        assert all(-1.0 <= c <= 1.0 for c in control), "This function accepts only controls between -1.0 and 1.0"
        if control[0] > 0:  # gas
            self.gamepad.right_trigger_float(value_float=control[0])
        else:
            self.gamepad.right_trigger_float(value_float=0.0)
        if control[1] > 0:  # break
            self.gamepad.left_trigger_float(value_float=control[1])
        else:
            self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.left_joystick_float(control[2], 0.0)  # turn
        self.gamepad.update()

    def reset(self):
        """
        Resetea todos los controles del gamepad.
        """
        self.gamepad.reset()
        self.gamepad.update()  # Actualizar el estado del gamepad



