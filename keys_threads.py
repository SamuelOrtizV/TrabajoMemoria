from pynput.keyboard import Key, Listener
import threading
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

def key_listener():
    # Start the listener in a separate thread
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()  # Wait for the listener to finish

def main():
    print("Press 'Q' to quit.")
    while True:
        keys = key_check()
        print(keys, end="\r")

        if keys == "Q":
            break

if __name__ == "__main__":
    # Start the key listener in a separate thread
    listener_thread = threading.Thread(target=key_listener)
    listener_thread.start()

    # Run the main function in the main thread
    main()

    # Stop the listener once the main loop is done
    listener_thread.join()  # Ensure listener_thread finishes
