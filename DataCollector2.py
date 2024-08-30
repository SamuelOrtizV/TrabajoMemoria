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

WIDTH = 192
HEIGHT = 144 #192x144 es apenas distinguible por el ojo humano, un buen punto de partida

file_name = "training_data.npz"
data_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data"

if os.path.isfile(file_name):
    print(f"Archivo {file_name} encontrado, cargando datos existentes...")
    training_data = list(np.load(file_name, allow_pickle=True)) # allow_pickle=True means that the data is loaded as a list
else:
    print(f"Archivo {file_name} no encontrado, creando uno nuevo...")
    training_data = []

def save_images_with_labels(images, labels, save_path, file_name):
    data = {'images': images, 'labels': labels}
    np.savez(os.path.join(save_path, file_name), **data)



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

    frame_count = 0
    images = []
    labels = []

    # Bucle principal
    while True:
        
        img = capture_screen(region)
        preprocessed_img = preprocess_image(img, WIDTH, HEIGHT)

        keys = key_check()
        output = keys_to_id(keys)

        images.append(preprocessed_img)
        labels.append(output)


        print(f"Keys: {keys} Output: {output}")        

        training_data.append([preprocessed_img, output])

        if len(images) % 10 == 0:
            print(f"Guardando datos de entrenamiento. Imagenes capturadas: {len(training_data)} ")
            save_images_with_labels(np.array(images), np.array(labels), data_path, file_name)
            images.clear()
            labels.clear()

        if args.show_screen_capture:
            if show_screen_capture(img):            
                break

if __name__ == "__main__":
    main()