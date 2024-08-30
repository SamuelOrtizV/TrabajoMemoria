import numpy as np
import cv2
import mss
import os

path= r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data\captured_image_001.npz"

def load_and_show_image(file_path):
    # Carga la imagen y la etiqueta desde el archivo npz
    data = np.load(file_path)
    img = data['image']
    label = data['label']

    print(label)
    
    # Muestra la imagen en una ventana
    cv2.imshow(f'Image with Label: {label}', img)
    
    # Espera hasta que se presione una tecla
    cv2.waitKey(0)
    
    # Cierra la ventana
    cv2.destroyAllWindows()

# Ejemplo de uso
if __name__ == "__main__":
    file_path = path  # Reemplaza con la ruta de tu archivo
    load_and_show_image(file_path)
