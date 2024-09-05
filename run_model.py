import argparse
import numpy as np
import os
import time
import cv2
from ScreenRecorder import capture_screen, show_screen_capture, preprocess_image
from getkeys import key_check, id_to_key
from DriveInputs import none, move_left, move_right, move_forward, move_back, move_left_forward, move_right_forward, move_left_back, move_right_back
import time
from model import SimpleRNN
import torch


# Dimensiones de la imagen que entra al modelo
WIDTH = 128 
HEIGHT = 72
# Definición de parámetros del modelo
input_size = (WIDTH, HEIGHT)  # 16:9 ratio
hidden_size = 64 # Número de neuronas en la capa oculta
output_size = 9 # Número de clases (W, A, S, D, WA, WD, SA, SD, NONE)
seq_len = 5 # Número de imágenes a considerar en la secuencia
batch_size = 2 # Número de secuencias a considerar en paralelo
num_epochs = 30 # Número de veces que se recorrerá el dataset
learning_rate = 0.001


def run_model(
        width: int = 1600,
        height: int = 900,
        full_screen: bool = False,
        show_screen_capture: bool = False,
        max_fps: int = 5,
        model_path: str = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\trained_models\model_epoch_1.pth",
) -> None:
    # print countdown 5 seconds
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    print("Comenzando a ejecutar el modelo. Presione P para pausarlo o Q para salir...")

     # Define la región de captura (x, y, width, height)
    if full_screen:
        region = {'left': 0, 'top': 0, 'width': width, 'height': height}
    else:
        region = {'left': 0, 'top': 40, 'width': width, 'height': height}

    print(f"Capturando una región de {width}x{height} píxeles...")

    # Cargar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model.to(device)


    # Bucle principal
    sequence_buffer = []
    cont = 0
    try:
        while True:        
            start_time = time.time()

            img = capture_screen(region)            
            preprocessed_img = preprocess_image(img, WIDTH, HEIGHT)

            # Añadir la imagen preprocesada al buffer de secuencia
            sequence_buffer.append(preprocessed_img)
        
            # Mantener solo las últimas SEQUENCE_LENGTH imágenes en el buffer
            if len(sequence_buffer) > seq_len:
                sequence_buffer.pop(0)

            # Verificar si tenemos suficientes imágenes para una secuencia completa
            if len(sequence_buffer) == seq_len:

                # Convertir el buffer de secuencia a un numpy.ndarray y luego a un tensor de PyTorch
                sequence_array = np.array(sequence_buffer)
                sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to('cuda')

                # Asegurarse de que el tensor tiene la forma correcta [batch_size, seq_len, channels, height, width]
                if sequence_tensor.ndim == 4:
                    sequence_tensor = sequence_tensor.unsqueeze(2)  # Añadir dimensión de canales

                # Habilitar precisión mixta
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(sequence_tensor)

                # Obtener la predicción del modelo
                prediction = torch.argmax(output, dim=1).item()

                print(f"{cont} Modelo funcionando! Predicción del modelo: {id_to_key(prediction)}", end="\r")

                cont += 1

                if cont >= 10:
                    cont = 0

                """ if prediction == 0:
                    none()
                elif prediction == 1:
                    move_left()
                elif prediction == 2:
                    move_right()
                elif prediction == 3:
                    move_forward()
                elif prediction == 4:
                    move_back()
                elif prediction == 5:
                    move_left_forward()
                elif prediction == 6:
                    move_right_forward()
                elif prediction == 7:
                    move_left_back()
                elif prediction == 8:
                    move_right_back() """

            keys = key_check()

            if keys == "Q":
                raise KeyboardInterrupt
            elif keys == "P":
                print("Se ha pausado el modelo. Presione 'P' nuevamente para reanudar.", end="\r")
                while key_check() != "P":
                    time.sleep(0.1)
                time.sleep(1)
                print("                                                                                            ", end="\r")

            """ if show_screen_capture:
                cv2.imshow("Screen Capture", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break """

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / max_fps:
                time.sleep(1 / max_fps - elapsed_time)

    except KeyboardInterrupt:
        print("\nInterrupción de teclado")
        
    
    print("\nSaliendo del modelo...")

if __name__ == "__main__":

    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Ejecutar el modelo de conducción autónoma.")
    parser.add_argument("--width", type=int, default=1600, help="Ancho de la región de captura.")
    parser.add_argument("--height", type=int, default=900, help="Alto de la región de captura.")
    parser.add_argument("--full_screen", type=bool, default=False, help="Capturar pantalla completa.")
    parser.add_argument("--show_screen_capture", type=bool, default=False, help="Mostrar captura de pantalla.")
    parser.add_argument("--max_fps", type=int, default=5, help="Máximo número de fotogramas por segundo.")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\trained_models\model_epoch_1.pth", help="Ruta del modelo entrenado.")

    args = parser.parse_args()

    print("Iniciando...")

    run_model(
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        show_screen_capture=args.show_screen_capture,
        max_fps=args.max_fps,
        model_path=args.model_path
    )

        
