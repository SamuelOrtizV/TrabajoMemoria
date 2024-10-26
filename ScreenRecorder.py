import mss
import cv2
import numpy as np


def capture_screen(region):
    with mss.mss() as sct:
        # Captura la imagen de la región especificada
        img = sct.grab(region)
        frame = np.array(img)        

        return frame
    
def show_screen_capture(img):    
    cv2.imshow("Screen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return True

def get_region(screen_size, full_screen):
    
    if full_screen:
        region = {'left': 0, 'top': 0, 'width': screen_size[0], 'height': screen_size[1]}
    else:
        region = {'left': 0, 'top': 40, 'width': screen_size[0], 'height': screen_size[1]}

    return region

def preprocess_image(img, width, height):
    """
    Preprocesa la imagen capturada.

    :param img: Imagen capturada.
    :param width: Ancho de la imagen.
    :param height: Altura de la imagen.
    :return: Imagen preprocesada.
    """

    processed_image = cv2.resize(img, (width, height))

    return np.asarray(
        processed_image,
        dtype=np.uint8,
    )
