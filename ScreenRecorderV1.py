from PIL import ImageGrab
import numpy as np
import cv2
import time

from getkeys import key_check

# Get screenshots of the screen

last_time = time.time()

while(True):
    # Capture the screen
    img = ImageGrab.grab(bbox=(0,40,800,640)) # Get the screen image
    img_np = np.array(img) # Convert the image to a numpy array
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Convert the image to RGB
    #cv2.imshow("Screen", frame) # Display the screen
    #print("Recorded FPS: {}".format(1 / (time.time() - last_time))) # Calculate the frames per second
    print("Frame time: {}".format((time.time() - last_time))) # shows the time it takes to process the frame
    #print("Keys: {}".format(key_check())) # Check if any keys are pressed
    last_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quitq
        break