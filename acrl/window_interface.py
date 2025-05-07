import mss
import numpy as np
from tmrl.custom.tm.utils.window import WindowInterface
import platform

if platform.system() == "Windows":

    import win32gui

    class MSSWindowInterface(WindowInterface):
        def __init__(self, window_name):
            super().__init__(window_name)
            self.sct = mss.mss()

        def screenshot(self):
            """
            Captura la pantalla usando mss
            """
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."

            # Obtener las dimensiones de la ventana
            x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
            # Ajustar las dimensiones para excluir los bordes y la barra de título
            monitor = {
            "top": y + 32,
            "left": x + self.w_diff // 2, 
            "width": x1 - x - self.w_diff,
            "height": y1 - y - 40
            }

            # Capturar la imagen con mss
            img = np.array(self.sct.grab(monitor))
            return img

else:
    print("Please use class WindowsInterface directly for non-Windows platforms.")