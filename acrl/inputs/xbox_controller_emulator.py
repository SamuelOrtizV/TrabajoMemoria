"""Xbox 360 gamepad emulator utilities using vgamepad.

Provides a thin wrapper to send analog and button events to a virtual
Xbox controller. Designed for simple 2-control mode:
- control[0]: throttle/brake in [-1.0, 1.0] (>=0 throttle, <0 brake)
- control[1]: steering in [-1.0, 1.0]
"""

import vgamepad as vg
import time


class XboxControllerEmulator:
    """Minimal helper around vgamepad's VX360Gamepad.

    Methods:
      - control_gamepad(control): apply throttle/brake/steer values
      - next_gear(): tap B button to upshift
      - reset(): reset controller state
    """

    def __init__(self):
        # Create the virtual Xbox gamepad
        self.gamepad = vg.VX360Gamepad()
        self.wait_before_start = 1  # seconds before starting to avoid misreads

        # Wait a few seconds before starting to emulate the controller
        for delay in range(int(self.wait_before_start), 0, -1):
            print(
                f"Initializing controller emulation, waiting {delay} seconds...",
                end="\r",
            )
            time.sleep(1)

    def control_gamepad(self, control):
        """Apply analog controls to the virtual gamepad.

        control: iterable of floats in [-1.0, 1.0]
          index 0: throttle/brake (>=0 throttle, <0 brake)
          index 1: steering
        """
        assert all(-1.0 <= c <= 1.0 for c in control), (
            "This function accepts only controls between -1.0 and 1.0"
        )

        # 2-control mode: merge throttle and brake into one
        if control[0] >= 0:  # throttle
            self.gamepad.right_trigger_float(value_float=control[0])
            self.gamepad.left_trigger_float(value_float=0.0)
        else:  # brake
            self.gamepad.left_trigger_float(value_float=-control[0])
            self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_joystick_float(control[1], 0.0)  # steering

        self.gamepad.update()

    def next_gear(self):
        """Tap the B button to shift up."""
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()
        time.sleep(0.05)
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()

    def reset(self):
        """Reset all controller state and update the device."""
        self.gamepad.reset()
        self.gamepad.update()



