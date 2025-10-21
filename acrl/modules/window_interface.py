"""Window capture interface using mss for Windows.

Provides an implementation of TMRL's WindowInterface that grabs frames either
in fullscreen or cropped to exclude borders/title bar.
"""

import mss
import numpy as np
from tmrl.custom.tm.utils.window import WindowInterface
import platform

if platform.system() == "Windows":

    import win32gui

    class MSSWindowInterface(WindowInterface):
        def __init__(self, window_name, fullscreen = False):
            super().__init__(window_name)
            self.sct = mss.mss()
            self.fullscreen = fullscreen

        def screenshot(self):
            """Capture the window content using mss."""
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."

            # Window bounds
            x, y, x1, y1 = win32gui.GetWindowRect(hwnd)

            if self.fullscreen:
                monitor = {
                "top": y,
                "left": x, 
                "width": x1 - x,
                "height": y1 - y
                }
            else:
                # Adjust to exclude borders and title bar
                monitor = {
                "top": y + 32,
                "left": x + self.w_diff // 2, 
                "width": x1 - x - self.w_diff,
                "height": y1 - y - 40
                }

            img = np.array(self.sct.grab(monitor))
            return img

else:
    print("Please use class WindowsInterface directly for non-Windows platforms.")