"""
Este codigo se encargara de realizar la data augmentation de las imagenes capturadas.
El metodo de data augmentation que se utilizara sera el de aplicar un efecto espejo horizontal de la imagen junto al correspondiente cambio en la etiqueta.
"""

from DataInfo import get_data, sort_files, get_data_label, get_last_image_number
import cv2
import os
import time
import cupy as cp

source_path = "./datasets/silverstone67"
destination_path = "./datasets/silverstone67_4"

if not os.path.exists(destination_path):
    os.makedirs(destination_path)

def flip_images(files, source_path, save_path):
    """
    Realiza un efecto espejo horizontal de las imágenes y las guarda en la carpeta especificada.

    :param files: Lista con la información de las imágenes.
    :param save_path: Ruta donde se guardarán las imágenes.
    """
    sorted_files = sort_files(files)
    img_id = get_last_image_number(source_path)
    size = len(sorted_files)
    cont = 0

    for file in sorted_files:
        image = cv2.imread(os.path.join(source_path, file))
        if image is None:
            print(f"Advertencia: No se pudo leer la imagen {file}.")
            continue

        # Convertir la imagen a un array de CuPy
        image_gpu = cp.asarray(image)

        # Realizar el efecto espejo horizontal en la GPU
        image_flipped_gpu = cp.flip(image_gpu, axis=1)

        # Convertir la imagen de vuelta a un array de NumPy
        image_flipped = cp.asnumpy(image_flipped_gpu)

        steering, throttle = get_data_label(file)

        cv2.imwrite(os.path.join(save_path, f"{img_id}_{-steering} {throttle}.jpeg"), image_flipped) # Guardar imagen

        img_id += 1
        cont += 1

        if img_id % 100 == 0:
            print(f"Progreso: {cont}/{size}", end="\r")

print("Iniciando Data Augmentation...")
start_time = time.time()

files = get_data(source_path)   # Obtener la información de las imágenes
flip_images(files, source_path, destination_path)  # Realizar el efecto espejo horizontal

end_time = time.time() - start_time
end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))

print(f"\nData Augmentation finalizado en {end_time}.")