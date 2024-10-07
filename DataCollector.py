import argparse
import numpy as np
import os
import time
import cv2
from ScreenRecorder import capture_screen, show_screen_capture, preprocess_image
from inputs.getkeys import key_check, keys_to_id
import threading

# IDEA, SEPARAR LAS ENTRADAS EN DOS GRADOS DE LIBERTAD. GUARDAR LOS PRIMEDIOS DE LAS LECTURAS DE CADA GRADO DE LIBERTAD CAPTURADO EN UNA IMAGEN

# Define the size of the screen capture
WIDTH = 426
HEIGHT = 240

data_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\raw_data_test"
# Verificar si el directorio existe, si no, lo crea
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Directorio {data_path} creado.")

# Variable global para controlar la interrupción del teclado
stop_event = threading.Event()
pause_event = threading.Event()
delete_event = threading.Event()

def key_detection():
    global stop_event
    while not stop_event.is_set():
        keys = key_check()
        if keys == "Q":
            stop_event.set()
            #print("\nCaptura de datos terminada por el usuario.                                   \n")
        elif keys == "P":
            if pause_event.is_set():
                pause_event.clear()
                print("Reanudando el modelo...", end="\r")
            else:
                print("                                                                                            ", end="\r")
                pause_event.set()
            time.sleep(1)  # Evitar múltiples detecciones rápidas
        elif keys == "E":
            delete_event.set()
            time.sleep(0.5)


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
    :return: Número de la última imagen guardada y la cantidad de imágenes en la carpeta.
    """
    files = os.listdir(save_path)
    files = [int(f.split("_")[0]) for f in files if f.endswith(".jpeg")]

    dataset_size = len(files)

    if dataset_size == 0:
        return 0, 0

    files.sort()

    next_img_number = max(files) + 1

    return next_img_number, dataset_size

def delete_last_images(data_path, last_img_id, num_files):
    deleted_files = 0
    i = 0
    
    while deleted_files < num_files and last_img_id - i >= 0:
        for j in range(9):
            file_path = os.path.join(data_path, f"{last_img_id-i}_{j}.jpeg")
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files += 1
                if deleted_files >= num_files:
                    break
        i += 1

def data_collector(
        width: int = 1600,
        height: int = 900,
        full_screen: bool = False,
        show_screen_capture: bool = False,
        max_fps: int = 5,
        data_path: str = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\datasets\raw_data_test"
) -> None:
    """
    Captura la pantalla y guarda las imágenes en una carpeta con la etiqueta correspondiente a las teclas presionadas.

    :param width: Ancho de la región a capturar.
    :param height: Altura de la región a capturar.
    :param full_screen: Captura toda la pantalla o una ventana.
    :param show_screen_capture: Muestra la grabación de la pantalla.
    :param max_fps: Máximo número de fotogramas por segundo.
    :param data_path: Ruta donde se guardarán las imágenes.

    Presione 'Q' para detener la captura de datos.
    Presione 'P' para pausar la captura de datos.
    Presione 'E' para eliminar las últimas imágenes capturadas (En caso de error).
    """

    # Define la región de captura (x, y, width, height)
    if full_screen:
        region = {'left': 0, 'top': 0, 'width': width, 'height': height}
    else:
        region = {'left': 0, 'top': 40, 'width': width, 'height': height}

    print(f"Capturando una región de {width}x{height} píxeles...")

    # print countdown 5 seconds
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    img_id, dataset_size = get_last_image_number(data_path)
    #delete_count = 0

    print("Comenzando captura de datos a partir de la imagen", img_id)

    # Iniciar el hilo de detección de teclas
    key_thread = threading.Thread(target=key_detection)
    key_thread.start()

    # Bucle principal
    # run_app = True

    while not stop_event.is_set():
        try:
            start_time = time.time()

            # Pausar el bucle si pause_event está establecido
            if pause_event.is_set():
                print("Programa pausado. Presione 'P' para reanudar.                                                                  ", end="\r")
                time.sleep(0.1)
                continue            

            img = capture_screen(region)
            # INCLUIR LA TRANSFORMACIÓN DE IMG EN NUMPY ARRAY DENTRI DE LA FUNCIÓN preprocess_image
            preprocessed_img = preprocess_image(img, WIDTH, HEIGHT)

            keys = key_check()
            output = keys_to_id(keys)

            """ if keys == "Q":
                raise KeyboardInterrupt
            
            elif keys == "P":
                print("La captura de datos se encuentra en pausa. Presione 'P' para reanudar.                               ", end="\r")                
                while key_check() != "P":
                    time.sleep(0.1)
                time.sleep(1)

            elif keys == "E":
                print("Se han eliminado las últimas imágenes capturadas.")
                num_imgs_to_delete = max_fps * 2
                last_img_id = get_last_image_number(data_path)
                delete_last_images(data_path, last_img_id, num_imgs_to_delete)
                print(f"Se han eliminado las últimas {num_imgs_to_delete} imágenes capturadas.")
                continue """
            
            if delete_event.is_set():
                #print("Se han eliminado las últimas imágenes capturadas.")
                num_imgs_to_delete = max_fps * 2
                #delete_count += num_imgs_to_delete

                #last_img_id, dataset_size = get_last_image_number(data_path)

                delete_last_images(data_path, img_id, num_imgs_to_delete)

                img_id, dataset_size = get_last_image_number(data_path)
                
                print(f"Se han eliminado las últimas {num_imgs_to_delete} imágenes capturadas. Datos guardados: {dataset_size}.                              ")
                delete_event.clear()
                continue
                

            #print(f"Keys: {keys} Output: {output}")

            save_images_with_labels(preprocessed_img, output, data_path, img_id)
            img_id += 1
            dataset_size += 1

            wait_time = 1.0 / max_fps - (time.time() - start_time)
            if wait_time > 0:
                time.sleep(wait_time)

            """ if show_screen_capture:
                if show_screen_capture(img): #Esta parte requiere uso de hilos, NO USAR de momento           
                    break """

            print(f"Guardando {max_fps} imágenes por segundo. Datos guardados: {dataset_size}. Entrada teclado: {output}", end="\r")

        except KeyboardInterrupt:
            #run_app = False
            stop_event.set()
            print("\nCaptura de datos terminada por el usuario.                                   \n")
    print("\nSaliendo del programa...")
    key_thread.join()
    time.sleep(1)   

if __name__ == "__main__":

    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Genera datos de entrenamiento con capturas de pantalla y etiquetas de teclas presionadas.")
    parser.add_argument('--width', type=int, required=True, help='Ancho de la región a capturar')
    parser.add_argument('--height', type=int, required=True, help='Altura de la región a capturar')
    parser.add_argument('--max_fps', type=int, default=5, required=False, help='Máximo número de fotogramas por segundo')
    parser.add_argument('--full_screen', type=bool, default=False, required=False, help='Captura toda la pantalla o una ventana') 
    parser.add_argument('--show_screen_capture', type=bool, default=False, required=False, help='Muestra la grabación de la pantalla')
    parser.add_argument('--data_path', type=str, default=r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\datasets\raw_data_test", required=False, help='Ruta donde se guardarán las imágenes')

    args = parser.parse_args()

    data_collector(
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        show_screen_capture=args.show_screen_capture,
        max_fps=args.max_fps,
        data_path=args.data_path
    )