import argparse
import numpy as np
import os
import time
import cv2
from ScreenRecorder import capture_screen, show_screen_capture, preprocess_image
from getkeys import key_check, keys_to_id

# Define the size of the screen capture
""" WIDTH = 480
HEIGHT = 270 """

""" WIDTH = 192
HEIGHT = 144 """ #192x144 es apenas distinguible por el ojo humano, un buen punto de partida

WIDTH = 800
HEIGHT = 600

file_name = "training_data.npz"
data_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data"

if os.path.isfile(file_name):
    print(f"Archivo {file_name} encontrado, cargando datos existentes...")
    training_data = list(np.load(file_name, allow_pickle=True)) # allow_pickle=True means that the data is loaded as a list
else:
    print(f"Archivo {file_name} no encontrado, creando uno nuevo...")
    training_data = []

def save_images_with_labels(image, label, save_path, id):
    """
    Guarda cada imagen individualmente en formato .jpg con un nombre que incluye
    un número secuencial y la etiqueta correspondiente.

    :param images: Lista o array de imágenes.
    :param labels: Lista de etiquetas correspondientes a las imágenes.
    :param save_path: Ruta donde se guardarán las imágenes.
    :param start_number: Número a partir del cual iniciar el contador para el nombre de los archivos.
    """
    file_name = f"{id}_{label}.jpeg"

    file_path = os.path.join(save_path, file_name)

    cv2.imwrite(file_path, image)

def get_last_image_number(save_path):
    """
    Obtiene el número de la última imagen guardada en una carpeta.

    :param save_path: Ruta donde se guardarán las imágenes.
    :return: Número de la última imagen guardada.
    """
    files = os.listdir(save_path)
    files = [f for f in files if f.endswith(".jpeg")]

    """ if len(files) == 0:
        return 0

    files.sort()

    last_file = files[-1]
    number = int(last_file.split("_")[0])

    return number """

    return len(files)


def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Captura una región de la pantalla y la muestra en tiempo real.")
    parser.add_argument('--width', type=int, required=True, help='Ancho de la región a capturar')
    parser.add_argument('--height', type=int, required=True, help='Altura de la región a capturar')
    parser.add_argument('--full_screen', type=bool, default=False, required=False, help='Captura toda la pantalla o una ventana') 
    parser.add_argument('--show_screen_capture', type=bool, default=False, required=False, help='Muestra la grabación de la pantalla') 
    # make this false by default

    args = parser.parse_args()

    # Define la región de captura (x, y, width, height)
    if args.full_screen:
        region = {'left': 0, 'top': 0, 'width': args.width, 'height': args.height}
    else:
        region = {'left': 0, 'top': 40, 'width': args.width, 'height': args.height}

    print(f"Capturando una región de {args.width}x{args.height} píxeles...")

    # print countdown 5 seconds
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    img_id = get_last_image_number(data_path) + 1

    print("Comenzando captura de datos a partir de la imagen", img_id)

    # Bucle principal

    cont = 0
    while True:
        
        img = capture_screen(region)
        preprocessed_img = preprocess_image(img, WIDTH, HEIGHT)

        keys = key_check()
        output = keys_to_id(keys)

        print(f"Keys: {keys} Output: {output}")

        save_images_with_labels(preprocessed_img, output, data_path, img_id)
        img_id += 1

        if args.show_screen_capture:
            if show_screen_capture(img):            
                break

        cont += 1

        if cont >=50:
            break

        

if __name__ == "__main__":
    main()