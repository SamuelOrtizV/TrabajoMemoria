import numpy as np
import cv2
import mss
import os

def capture_screen(region):
    with mss.mss() as sct:
        img = sct.grab(region)
        frame = np.array(img)        
        return frame
    
def preprocess_image(img, width, height):
    bnw_frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    processed_image = cv2.resize(bnw_frame, (width, height))
    return processed_image

def save_image_with_label(img, label, save_path, file_name):
    # Combina la imagen y la etiqueta en un diccionario
    data = {'image': img, 'label': label}
    # Guarda el diccionario en un archivo npz
    np.savez(os.path.join(save_path, file_name), **data)

# Ejemplo de uso
if __name__ == "__main__":
    region = {"top": 100, "left": 100, "width": 800, "height": 600}
    save_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data"
    label = 1  # Ejemplo de etiqueta

    img = capture_screen(region)
    processed_img = preprocess_image(img, 800, 600)  # Ajusta el tamaño según lo necesites
    save_image_with_label(processed_img, label, save_path, "captured_image_001.npz")
