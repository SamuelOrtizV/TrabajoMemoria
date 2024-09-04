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

WIDTH = 128
HEIGHT = 72 #128 x 72 es apenas distinguible por el ojo humano, un buen punto de partida

""" WIDTH = 1600
HEIGHT = 900 """

data_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\raw_data"

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
    files = [int(f.split("_")[0]) for f in files if f.endswith(".jpeg")]

    if len(files) == 0:
        return -1

    files.sort()

    return max(files)

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
        data_path: str = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data"
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

    img_id = get_last_image_number(data_path) + 1

    print("Comenzando captura de datos a partir de la imagen", img_id)

    # Bucle principal

    run_app = True

    while run_app:
        try:
            start_time = time.time()            

            img = capture_screen(region)
            preprocessed_img = preprocess_image(img, WIDTH, HEIGHT)

            keys = key_check()
            output = keys_to_id(keys)

            if keys == "Q":
                raise KeyboardInterrupt
            elif keys == "P":
                print("Se ha pausado la captura de datos. Presione 'P' nuevamente para reanudar.", end="\r")
                time.sleep(1)
                while key_check() != "P":
                    time.sleep(0.1)
            elif keys == "E":
                print("Se han eliminado las últimas imágenes capturadas.")
                num_imgs_to_delete = max_fps * 2
                last_img_id = get_last_image_number(data_path)
                delete_last_images(data_path, last_img_id, num_imgs_to_delete)
                print(f"Se han eliminado las últimas {num_imgs_to_delete} imágenes capturadas.")
                continue
                

            #print(f"Keys: {keys} Output: {output}")

            save_images_with_labels(preprocessed_img, output, data_path, img_id)
            img_id += 1

            wait_time = 1.0 / max_fps - (time.time() - start_time)
            if wait_time > 0:
                time.sleep(wait_time)

            """ if show_screen_capture:
                if show_screen_capture(img): #Esta parte requiere uso de hilos, NO USAR de momento           
                    break """
            
            print(f"Guardando {max_fps} imágenes por segundo. Datos guardados: {img_id-1}. Entrada teclado: {output}", end="\r")

        except KeyboardInterrupt:
            run_app = False
            print("\nCaptura de datos interrumpida por el usuario.\n"
                  "Se han guardado un total de", img_id-1, "imágenes.")    

if __name__ == "__main__":

    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Genera datos de entrenamiento con capturas de pantalla y etiquetas de teclas presionadas.")
    parser.add_argument('--width', type=int, required=True, help='Ancho de la región a capturar')
    parser.add_argument('--height', type=int, required=True, help='Altura de la región a capturar')
    parser.add_argument('--max_fps', type=int, default=5, required=False, help='Máximo número de fotogramas por segundo')
    parser.add_argument('--full_screen', type=bool, default=False, required=False, help='Captura toda la pantalla o una ventana') 
    parser.add_argument('--show_screen_capture', type=bool, default=False, required=False, help='Muestra la grabación de la pantalla')
    parser.add_argument('--data_path', type=str, default=r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\data", required=False, help='Ruta donde se guardarán las imágenes')

    args = parser.parse_args()

    data_collector(
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        show_screen_capture=args.show_screen_capture,
        max_fps=args.max_fps,
        data_path=args.data_path
    )