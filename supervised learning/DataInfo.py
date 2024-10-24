import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
from inputs.getkeys import id_to_key
from PIL import Image
import io

data_path_source = "./datasets/test3"
data_path_destination = "./datasets/output_data"

if not os.path.exists(data_path_destination):
    os.makedirs(data_path_destination)

DATA_FRAME_SIZE = 5

def get_data(save_path_source):
    """
    Obtiene la información de las imágenes guardadas en una carpeta.

    :param save_path: Ruta donde se guardarán las imágenes.
    :return: Lista con la información de las imágenes.
    """
    files = os.listdir(save_path_source)
    files = [f for f in files if f.endswith(".jpeg")]

    return files

def sort_files(files):
    """
    Ordena los archivos de una carpeta en base al primer numero del nombre.

    :param files: Lista con la información de las imágenes.
    :return: Lista con la información de las imágenes ordenadas.
    """
    files.sort(key=lambda x: int(x.split("_")[0]))

    return files
  

def get_data_label(file):
    """
    Obtiene las etiquetas de las imágenes guardadas en una carpeta.

    :param files: Lista con la información de las imágenes.
    :return: Lista con las etiquetas de las imágenes.
    """
    label = file.split("_")[1].rsplit('.', 1)[0]

    steering = float(label.split(" ")[0])
    throttle = float(label.split(" ")[1])

    return steering, throttle

def get_data_labels(files):
    """
    Obtiene las etiquetas de las imágenes guardadas en una carpeta.

    :param files: Lista con la información de las imágenes.
    :return: Lista con las etiquetas de las imágenes.
    """
    labels = [f.split("_")[1].rsplit('.', 1)[0] for f in files]

    steering = [float(label.split(" ")[0]) for label in labels]
    throttle = [float(label.split(" ")[1]) for label in labels]

    return steering, throttle

def get_last_image_number(save_path):
    """
    Obtiene el número de la última imagen guardada en una carpeta.

    :param save_path: Ruta donde se guardarán las imágenes.
    :return: Número de la última imagen guardada y la cantidad de imágenes en la carpeta.
    """
    files = os.listdir(save_path)
    files = [int(f.split("_")[0]) for f in files if f.endswith(".jpeg")]

    dataset_size = len(files)

    if dataset_size == 0:
        return 0

    files.sort()

    next_img_number = max(files) + 1

    return next_img_number

def show_histogram(labels):
    """
    Muestra un histograma con la cantidad de imágenes por etiqueta.

    :param labels: Lista con las etiquetas de las imágenes.
    """

    data = np.array(labels)
    bins = np.linspace(-1, 1, 21)

    frecuencia, bordes = np.histogram(data, bins=bins)

    # Graficar el histograma
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title("Distribución de frecuencias")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.show()

def get_labels_count(labels):
    """
    Obtiene la cantidad de imágenes por etiqueta.

    :param labels: Lista con las etiquetas de las imágenes.
    :return: Diccionario con la cantidad de imágenes por etiqueta.
    """

    keys = [id_to_key(label) for label in labels]

    labels_count = {}
    for label in keys:
        if label in labels_count:
            labels_count[label] += 1
        else:
            labels_count[label] = 1

    return labels_count

def get_label_percentage(labels_count):
    """
    Obtiene el porcentaje de imágenes por etiqueta.

    :param labels_count: Diccionario con la cantidad de imágenes por etiqueta.
    :return: Diccionario con el porcentaje de imágenes por etiqueta.
    """
    total = sum(labels_count.values())

    labels_percentage = {}
    for label, count in labels_count.items():
        labels_percentage[label] = round(count / total * 100,1)

    return labels_percentage

def group_and_shuffle_files(files, group_size):
    """
    Agrupa los archivos en grupos de tamaño `group_size`, mezcla aleatoriamente esos grupos y actualiza la ID de cada imagen.

    :param files: Lista con la información de las imágenes.
    :param group_size: Tamaño de cada grupo.
    :return: Lista con los archivos agrupados, mezclados y con IDs actualizadas.
    """
    # Dividir en grupos
    groups = [files[i:i + group_size] for i in range(0, len(files), group_size)]
    
    # Mezclar los grupos
    random.shuffle(groups)
    
    # Actualizar la ID de cada imagen
    updated_files = []
    for group_index, group in enumerate(groups):
        for file in group:
            parts = file.split("_")
            parts[0] = str(group_index * group_size + group.index(file))
            updated_file = "_".join(parts)
            updated_files.append(updated_file)
    
    return updated_files


# ---------------------------------------Función provisoria-------------------------------------------------
def get_balanced_data(sorted_files, labels):

    """
    Obtiene una cantidad balanceada de imágenes por etiqueta.

    :param sorted_files: Lista con las imágenes ordenadas.
    :param labels: Lista con las etiquetas de las imágenes.
    """
    
    balanced_files = []

    for i in range(math.floor(len(labels)/DATA_FRAME_SIZE)):
        data_frame = []
        flag = False
        for j in range(DATA_FRAME_SIZE):
            index = i*DATA_FRAME_SIZE+j
            data_frame.append(sorted_files[index])

            if labels[index] != 3:
                flag = True

        if flag: # 
            balanced_files.extend(data_frame) # Agregar data_frame a balanced_files
        else:
            # Tener un 10% de probabilidad de agregar data_frame a balanced_files para los casos de "W"
            if random.random() < 0.1:
                balanced_files.extend(data_frame)

    return balanced_files  

def get_recomended_weights(labels_percentage):
    """
    Obtiene los pesos recomendados para cada etiqueta.

    :param labels_percentage: Diccionario con el porcentaje de imágenes por etiqueta.
    :return: Diccionario con los pesos recomendados para cada etiqueta.
    """
    total = sum(labels_percentage.values())
    weights = {}
    
    for label, count in labels_percentage.items():
        if count == 0:
            weights[label] = 0  # Asignar un peso de 0 si el porcentaje es 0
        else:
            weights[label] = round(total / (count * 100), 2)

    return weights

def preprocess_image(img, width, height, quality=90):
    """
    Redimensiona y comprime una imagen.

    :param img: Imagen a redimensionar y comprimir.
    :param width: Ancho de la imagen redimensionada.
    :param height: Alto de la imagen redimensionada.
    :param quality: Calidad de la compresión JPEG.
    :return: Imagen redimensionada y comprimida.
    """

    processed_image = cv2.resize(img, (width, height))
    pil_image = Image.fromarray(processed_image)

    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    compressed_image = Image.open(buffer)
    compressed_image_np = np.asarray(compressed_image, dtype=np.uint8)

    return compressed_image_np

def resize_images(files, source_path, destination_path, width, height, quality=70) -> None:
    print("Redimensionando y comprimiendo imágenes...")
    for file in files:
        source = os.path.join(source_path, file)
        destination = os.path.join(destination_path, file)
        img = cv2.imread(source)
        if img is not None:
            img = preprocess_image(img, width, height, quality)
            cv2.imwrite(destination, img)
    print("Redimension y compresión de imágenes completa.")
          
def save_data(files, destination_path, source_path) -> None:
    """
    Copia las imágenes seleccionadas de una carpeta a otra.

    :param files: Lista con las imágenes seleccionadas.
    :param destination_path: Ruta donde se guardarán las imágenes seleccionadas.
    :param source_path: Ruta donde se encuentran las imágenes.
    """

    for file in files:
        source = os.path.join(source_path, file)
        destination = os.path.join(destination_path, file)
        img = cv2.imread(source)
        cv2.imwrite(destination, img)

        
""" start_time = time.time()

files = get_data(data_path_source)
sorted_files = sort_files(files)
    
steering, throttle = get_data_labels(sorted_files)

show_histogram(throttle)
show_histogram(steering)

end_time = time.time() - start_time  # Tiempo de finalización del epoch
# Convertir end_time a formato hh:mm:ss
end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))

print(f"Tiempo total: {end_time}") """