import argparse
import numpy as np
import os
import time
import cv2
from ScreenRecorder import capture_screen, show_screen_capture, preprocess_image
from inputs.getkeys import key_check, id_to_key
from inputs.DriveInputs import move
from inputs.xbox_controller_emulator import XboxControllerEmulator
from torchvision import transforms, models
import time
from model import CNN_RNN, CNN_LSTM_STATE
import torch
from PIL import Image
import threading
import tkinter as tk

MODEL_NAME = "mini-CNN_LSTM_STATE-efficientnet_b0-20-240-135-2-256-epoch_1"+".pth"

# Model name format: {name}_{architecture}-{cnn_name}-{seq_len}-{input_size[0]}-{input_size[1]}-{hidden_size}-epoch_{epoch}.pth

""" cnn_name = "efficientnet_b0"
seq_len = 5
input_size = (224, 224)
hidden_size = 256 """

# Definición de parámetros
name = MODEL_NAME.split("-")[0]
architecture = MODEL_NAME.split("-")[1]
cnn_name = MODEL_NAME.split("-")[2]
seq_len = int(MODEL_NAME.split("-")[3])
input_size = (int(MODEL_NAME.split("-")[4]), int(MODEL_NAME.split("-")[5]))
output_size = int(MODEL_NAME.split("-")[6])
hidden_size = int(MODEL_NAME.split("-")[7])

print("Modelo a cargar:", MODEL_NAME)
print("Nombre:", name)
print("Arquitectura:", architecture)
print("CNN:", cnn_name)
print("Seq_len:", seq_len)
print("Input size:", input_size)
print("Output size:", output_size)
print("Hidden size:", hidden_size)

num_layers = 1
dropout = 0
bias = True

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((input_size)),  # Cambia el tamaño de las imágenes a (height x width)
    transforms.ToTensor(),         # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

if output_size == 9:
    print("Modelo de teclado cargado.")
elif output_size == 2:
    controller = XboxControllerEmulator()
    print("Modelo de controlador cargado.")
else:
    raise ValueError("El tamaño de salida del modelo debe ser 2 (control) o 9 (teclado).")


# Variable global para controlar la interrupción del teclado
stop_event = threading.Event()
pause_event = threading.Event()

def key_detection():
    global stop_event
    while not stop_event.is_set():
        keys = key_check()
        if keys == "Q":
            stop_event.set()
        elif keys == "P":
            if pause_event.is_set():
                pause_event.clear()
                print("Reanudando el modelo...", end="\r")
            else:
                print("                                                                                            ", end="\r")
                pause_event.set()
            time.sleep(1)  # Evitar múltiples detecciones rápidas
        elif keys == "W":
            if output_size == 2:
                controller.throttle_break(1.0)
                time.sleep(0.5)
                controller.reset()
            

def run_model(
        model_path: str,
        width: int = 1920,
        height: int = 1080,
        full_screen: bool = True,
        show_screen_capture: bool = False,
        max_fps: int = 10,
        
) -> None:
    # Crear la ventana de estado
    root = tk.Tk()
    var = tk.StringVar()
    var.set("Iniciando...")
    text_label = tk.Label(root, textvariable=var, fg="green", font=("Impact", 44))
    text_label.pack()
    
    print("Comenzando a ejecutar el modelo. Presione P para pausarlo o Q para salir...")

     # Define la región de captura (x, y, width, height)
    if full_screen:
        region = {'left': 0, 'top': 0, 'width': width, 'height': height}
    else:
        region = {'left': 0, 'top': 40, 'width': width, 'height': height}

    print(f"Capturando una región de {width}x{height} píxeles...")

    # Cargar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando", "cuda" if torch.cuda.is_available() else "cpu", "como dispositivo.\n\n\n\n\n")

    if architecture == "CNN_RNN":
        model = CNN_RNN(cnn_name = cnn_name, hidden_size = hidden_size, output_size = output_size,
                        input_size = (3, *input_size), num_layers = num_layers, dropout = dropout, bias = bias)
    elif architecture == "CNN_LSTM_STATE":
        model = CNN_LSTM_STATE(cnn_name = cnn_name, hidden_size = hidden_size, output_size = output_size,
                        input_size = (3, *input_size), num_layers = num_layers, dropout = dropout, bias = bias)
        hidden_state = model.init_hidden(1)  # Inicializar el hidden state con batch size = 1
        hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))  # Mover hidden_state al dispositivo
    else:
        raise ValueError("La arquitectura del modelo debe ser CNN_RNN o CNN_LSTM_STATE.")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model.to(device)

    # Iniciar el hilo de detección de teclas
    key_thread = threading.Thread(target=key_detection)
    key_thread.start()

    dots = ["   ", ".  ", ".. ", "...", " ..", "  ."]

    negative_throttle_time = 0
    invert_start_time = None
    invert_duration = 3  # Duración de la inversión en segundos
    stop_break_threshold = 5  # Tiempo en segundos para dejar de detener el vehículo

    sequence_buffer = []
    cont = 0
    # Bucle principal
    while not stop_event.is_set():
        try:                
            start_time = time.time()

            # Pausar el bucle si pause_event está establecido
            if pause_event.is_set():
                var.set("IA EN PAUSA")
                text_label.config(fg="red")
                root.update()
                if output_size == 2:
                    controller.reset()
                print("Modelo pausado. Presione 'P' para reanudar.                                                                  ", end="\r")
                time.sleep(0.1)
                continue
            else:
                var.set("IA FUNCIONANDO")
                text_label.config(fg="green")
                #root.update()

            img = capture_screen(region)        
            #preprocessed_img = preprocess_image(img, WIDTH, HEIGHT) # CREO QUE ESTA LINEA ES INNECESARIA
            # Convertir la imagen preprocesada a un objeto PIL y aplicar las transformaciones
            preprocessed_img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            preprocessed_img = transform(preprocessed_img)#.numpy() 


            # Añadir la imagen preprocesada al buffer de secuencia
            sequence_buffer.append(preprocessed_img)
        
            # Mantener solo las últimas imágenes en el buffer
            if len(sequence_buffer) > seq_len:
                sequence_buffer.pop(0)

            # Verificar si tenemos suficientes imágenes para una secuencia completa
            if len(sequence_buffer) == seq_len:
                # Convertir el buffer de secuencia a un numpy.ndarray y luego a un tensor de PyTorch
                #sequence_array = np.array(sequence_buffer)
                #sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to('cuda') # NO SE SI CONVERTIR DE NUEVO A TENSOR

                sequence_tensor = torch.stack(sequence_buffer).unsqueeze(0).to(device)

                #print("Tamaño secuence tensor", sequence_tensor.size())
                #print(type(sequence_tensor))

                moving_dots = dots[cont % len(dots)]

                # Habilitar precisión mixta
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    if architecture == "CNN_LSTM_STATE":
                        # Pasar el hidden_state y obtener la predicción y el nuevo hidden_state
                        output, hidden_state = model(sequence_tensor, hidden_state)
                        # Desconectar el hidden_state para evitar acumulación de gradientes
                        hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                    elif architecture == "CNN_RNN":
                        output = model(sequence_tensor)

                if output_size == 9: # Caso de teclado, discreto. Se usa argmax
                    # Obtener la predicción del modelo
                    prediction = torch.argmax(output, dim=1).item()

                    # Mover el vehículo
                    move(prediction)
                    print(f"{moving_dots} Modelo funcionando! Predicción del modelo: {id_to_key(prediction)}", end="\r")

                elif output_size == 2: # Caso de control, continuo. Se usa el valor directamente
                    prediction = torch.clamp(output, min=-1.0, max=1.0).tolist()[0] # Limitar los valores entre -1.0 y 1.0

                    if prediction[1] < 0:
                        negative_throttle_time += time.time() - start_time
                    else:
                        negative_throttle_time = 0

                    if negative_throttle_time > stop_break_threshold:
                        if invert_start_time is None:
                            invert_start_time = time.time()
                        if time.time() - invert_start_time < invert_duration:
                            prediction[1] = 1.0
                        else:
                            invert_start_time = None
                            negative_throttle_time = 0

                    controller.steering(prediction[0])
                    controller.throttle_break(prediction[1])
                    print(f"{moving_dots} Modelo funcionando! Predicción del modelo: Steering {prediction[0]:.2f} Throttle {prediction[1]:.2f}", end="\r")       

                cont += 1
                if cont > 5:
                    cont = 0

            """ if show_screen_capture:
                cv2.imshow("Screen Capture", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break """
            root.update_idletasks()
            root.update()

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / max_fps:
                time.sleep(1 / max_fps - elapsed_time)

        except KeyboardInterrupt:
            stop_event.set()
            print("\nInterrupción de teclado")
        except ValueError as e:
            stop_event.set()
            print(f"Error: {str(e)}")
        
    print("\nSaliendo del modelo...")   
    if output_size == 2:
        controller.reset()
    key_thread.join()
    root.quit()
    root.destroy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ejecutar el modelo de conducción autónoma.")
    parser.add_argument("--model_path", type=str, default=f"./trained_models/{name}/{MODEL_NAME}", help="Ruta del modelo entrenado.")
    parser.add_argument("--width", type=int, default=1920, help="Ancho de la región de captura.")
    parser.add_argument("--height", type=int, default=1080, help="Alto de la región de captura.")
    parser.add_argument("--full_screen", type=bool, default=True, help="Capturar pantalla completa.")
    parser.add_argument("--show_screen_capture", type=bool, default=False, help="Mostrar captura de pantalla.")
    parser.add_argument("--max_fps", type=int, default=10, help="Número de fotogramas por segundo.")
    
    args = parser.parse_args()

    print("Iniciando...")

    run_model(
        model_path=args.model_path,
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        show_screen_capture=args.show_screen_capture,
        max_fps=args.max_fps       
    )
