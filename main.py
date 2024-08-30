import numpy as np
#from grabscreen import grab_screen
from ScreenRecorder import capture_screen, show_screen_capture
import cv2
import time
from getkeys import key_check, keys_to_id
import os
import mss

#get random number from 0 to 9
random_number = np.random.randint(0,100)

file_name = str(random_number)+'.npy'

print(file_name)

# Define la región de captura (x, y, width, height)
region = {'left': 0, 'top': 40, 'width': 800, 'height': 600}

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def capture_screen(region):
    with mss.mss() as sct:
        # Captura la imagen de la región especificada
        img = sct.grab(region)
        frame = np.array(img)        
        return frame
    
def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True):
        img = capture_screen(region)
        keys = key_check()
        output = keys_to_id(keys)
        data = img.append(output)
        training_data.append(data)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 10 == 0:
            print(len(training_data))
            np.save(file_name,training_data)

main()