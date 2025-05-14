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
        
        # En caso de usar 3 controles, el primero es el acelerador, el segundo es el freno y el tercero es la dirección
        """ if control[0] > 0:  # gas
            self.gamepad.right_trigger_float(value_float=control[0])
        else:
            self.gamepad.right_trigger_float(value_float=0.0)
        if control[1] > 0:  # brake
            self.gamepad.left_trigger_float(value_float=control[1])
        else:
            self.gamepad.left_trigger_float(value_float=0.0)

        self.gamepad.left_joystick_float(control[2], 0.0) """
        
        # En caso de usar 2 controles, juntando acelerador y freno en uno solo
        if control[0] >= 0: #gas
            self.gamepad.right_trigger_float(value_float=control[0])
            self.gamepad.left_trigger_float(value_float=0.0)
        elif control[0] < 0: #brake
            self.gamepad.left_trigger_float(value_float=-control[0])
            self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_joystick_float(control[1], 0.0)  # turn

        self.gamepad.update()
    
    def next_gear(self):
        """
        Cambia a la siguiente marcha.
        """
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()
        time.sleep(0.05)  # Espera un poco para simular el tiempo de pulsación del botón
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()

    def reset(self):
        """
        Resetea todos los controles del gamepad.
        """
        self.gamepad.reset()
        self.gamepad.update()  # Actualizar el estado del gamepad



