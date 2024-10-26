from inputs.GameInputs import reset_environment, move_forward, move_back
import time

try:
    while True:
        reset_environment()
        move_forward()
        time.sleep(10)
        move_back()
except KeyboardInterrupt:
    print("Exiting...")
