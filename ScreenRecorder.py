import mss
import time
import cv2
import numpy as np

from getkeys import key_check


def capture_screen(region):
    with mss.mss() as sct:
        start_time = time.time()
        # Captura la imagen de la región especificada
        img = sct.grab(region)
        frame = np.array(img)        
        end_time = time.time()
        #print(f"Framerate: {end_time - start_time:.4f} segundos"+" Keys: {}".format(key_check()))
        #print() # Check if any keys are pressed
        return frame
    
# Show screen capture

def show_screen_capture(img):    
    cv2.imshow("Screen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return True
    

def preprocess_image(img, width, height):
    """
    Given an image resize it and convert it to a numpy array

    :param PIL.image image:
    :returns:
        numpy ndarray - image as a numpy array of dimensions [width, height, 1]
    """
    #bnw_frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    processed_image = cv2.resize(img, (width, height))

    #return processed_image

    return np.asarray(
        processed_image,
        dtype=np.uint8,
    )
    



# Define la región de captura (x, y, width, height)
region = {'left': 0, 'top': 40, 'width': 800, 'height': 600}

""" # Prueba Captura la pantalla
while True:
    img = capture_screen(region)
    if show_screen_capture(img):
        break
    # Aquí puedes procesar la imagen o guardarla si lo deseas
    # Por ejemplo, para guardarla:
    # mss.tools.to_png(img.rgb, img.size, output="screenshot.png")
 """