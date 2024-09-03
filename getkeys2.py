from pynput.keyboard import Key, Listener
import time

keys_pressed = []

def on_press(key):
    try:
        if key.char.upper() in "WASDQPE":
            keys_pressed.append(key.char.upper())
    except AttributeError:
        pass

def on_release(key):
    if key == Key.esc:
        # Stop listener
        return False

def key_check():
    global keys_pressed
    keys = list(set(keys_pressed))
    keys_pressed = []  # Reset after every check to avoid repeated keys
    return "".join(keys)

# Start the listener in a separate thread
listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Press 'Q' to quit.")
while True:
    keys = key_check()
    print(keys, end="\r")

    if keys == "Q":
        break

listener.stop()