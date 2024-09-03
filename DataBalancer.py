import os
import cv2
import numpy as np
import time
import random
import math
from getkeys import id_to_key

data_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data"
data_path_balanced = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\balanced_data"

DATA_FRAME_SIZE = 5

def get_data_info(save_path):
    """
    Obtiene la información de las imágenes guardadas en una carpeta.

    :param save_path: Ruta donde se guardarán las imágenes.
    :return: Lista con la información de las imágenes.
    """
    files = os.listdir(save_path)
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
  

def get_data_labels(files):
    """
    Obtiene las etiquetas de las imágenes guardadas en una carpeta.

    :param files: Lista con la información de las imágenes.
    :return: Lista con las etiquetas de las imágenes.
    """
    labels = [int(f.split("_")[1].split(".")[0]) for f in files]

    return labels

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
            # Tener un 10% de probabilidad de agregar data_frame a balanced_files
            if random.random() < 0.1:
                balanced_files.extend(data_frame)

    return balanced_files  
          
def save_data(balanced_files, destination_path, source_path) -> None:
    """
    Copia las imágenes seleccionadas de una carpeta a otra.

    :param balanced_files: Lista con las imágenes seleccionadas.
    :param destination_path: Ruta donde se guardarán las imágenes seleccionadas.
    :param source_path: Ruta donde se encuentran las imágenes.
    """

    for file in balanced_files:
        source = os.path.join(source_path, file)
        destination = os.path.join(destination_path, file)
        img = cv2.imread(source)
        cv2.imwrite(destination, img)




        

files = get_data_info(data_path)
sorted_files = sort_files(files)
labels = get_data_labels(sorted_files)
balanced_files = get_balanced_data(sorted_files, labels)
""" balanced_labels = get_data_labels(balanced_files)

labels_count = get_labels_count(labels)
balanced_labels_count = get_labels_count(balanced_labels)

labels_percentage = get_label_percentage(labels_count)
balanced_labels_percentage = get_label_percentage(balanced_labels_count)

print(labels_count)
print(balanced_labels_count)

print(labels_percentage)
print(balanced_labels_percentage) """

print("Balanceando los datos...", end="\r")

save_data(balanced_files, data_path_balanced, data_path)

print("Balance de datos completo. Datos balanceados guardados en la carpeta 'balanced_data'.")