# Crea un codigo que tome todas las imagenes .jpeg desde una dirección y las comprima a dimensiones 128 x 72 y las guarde en otra dirección

import cv2
import os

def compress_images(data_path, save_path, width=128, height=72):
    """
    Comprime las imágenes de una carpeta a un tamaño específico y las guarda en otra carpeta.

    :param data_path: Ruta donde se encuentran las imágenes originales.
    :param save_path: Ruta donde se guardarán las imágenes comprimidas.
    :param width: Ancho de las imágenes comprimidas.
    :param height: Alto de las imágenes comprimidas.
    """
    files = os.listdir(data_path)
    files = [f for f in files if f.endswith(".jpeg")]

    for file in files:
        img = cv2.imread(os.path.join(data_path, file))
        img = cv2.resize(img, (width, height))
        cv2.imwrite(os.path.join(save_path, file), img)

data_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\test_datasetr"
save_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\test_dataset"

os.makedirs(save_path, exist_ok=True)

print("Compressing images...")

compress_images(data_path, save_path)

print("Process finished")