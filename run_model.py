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
from model import SimpleRNN
import torch
from PIL import Image
import threading
import tkinter as tk

MODEL_NAME = "Model-efficientnet_b0-5-240-135-2-256-epoch_4.pth"
#MODEL_NAME = "Oval_Model_4_EffNet_b0_CNNGRAD_256_epoch_11.pth"
# Model name format: Model_{cnn_name}_{seq_len}_{input_size[0]}_{hidden_size}_epoch{epoch}.pth

""" cnn_name = "efficientnet_b0"
seq_len = 5
input_size = (224, 224)
hidden_size = 256 """


cnn_name = MODEL_NAME.split("-")[1]
seq_len = int(MODEL_NAME.split("-")[2])
input_size = (int(MODEL_NAME.split("-")[3]), int(MODEL_NAME.split("-")[4]))
output_size = int(MODEL_NAME.split("-")[5])
hidden_size = int(MODEL_NAME.split("-")[6])

print("Modelo a cargar:", MODEL_NAME)
print("CNN:", cnn_name)
print("Seq_len:", seq_len)
print("Input size:", input_size)
print("Output size:", output_size)
print("Hidden size:", hidden_size)

# Definición de parámetros
#pretrained_cnn = models.efficientnet_b0(weights='IMAGENET1K_V1') #models.efficientnet_v2_s(weights='IMAGENET1K_V1')
#hidden_size = 256 # Número de neuronas en la capa oculta 512 usado por Iker
# output_size = 9 # Número de clases (W, A, S, D, WA, WD, SA, SD, NONE)
# seq_len = 5 # Número de imágenes a considerar en la secuencia
num_layers = 1
dropout = 0
bias = True

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((input_size)),  # Cambia el tamaño de las imágenes a (height x width)
    transforms.ToTensor(),         # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# ARREGLAR ESTO---------------------------------------------------------------------------------------------------------------------
""" transform = transforms.Compose([
    transforms.Resize(input_size[0], interpolation=Image.BICUBIC),
    transforms.Pad((0, 0, 0, 0), fill=0, padding_mode='constant'),  # Rellenar para obtener el tamaño deseado
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) """

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
    print("Usando", "cuda" if torch.cuda.is_available() else "cpu", "como dispositivo.\n\n\n\n\n")

    model = SimpleRNN(cnn_name = cnn_name, hidden_size = hidden_size, output_size = output_size,
                       input_size = (3, *input_size), num_layers = num_layers, dropout = dropout, bias = bias)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model.to(device)

    if output_size == 9:
        print("Modelo de teclado cargado.")
    elif output_size == 2:
        controller = XboxControllerEmulator()
        print("Modelo de controlador cargado.")
    else:
        raise ValueError("El tamaño de salida del modelo debe ser 2 (control) o 9 (teclado).")


    # Iniciar el hilo de detección de teclas
    key_thread = threading.Thread(target=key_detection)
    key_thread.start()

    dots = ["   ", ".  ", ".. ", "...", " ..", "  ."]

    # Bucle principal
    sequence_buffer = []
    cont = 0
    while not stop_event.is_set():
        try:                
            start_time = time.time()

            # Pausar el bucle si pause_event está establecido
            if pause_event.is_set():
                var.set("CONTROL HUMANO")
                text_label.config(fg="red")
                root.update()
                if output_size == 2:
                    controller.reset()
                print("Modelo pausado. Presione 'P' para reanudar.                                                                  ", end="\r")
                time.sleep(0.1)
                continue
            else:
                var.set("CONTROL DE IA")
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
                sequence_array = np.array(sequence_buffer)
                sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to('cuda') # NO SE SI CONVERTIR DE NUEVO A TENSOR
                #print("Tamaño secuence tensor", sequence_tensor.size())
                #print(type(sequence_tensor))

                moving_dots = dots[cont % len(dots)]

                # Habilitar precisión mixta
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(sequence_tensor)

                if output_size == 9: # Caso de teclado, discreto. Se usa argmax
                    # Obtener la predicción del modelo
                    prediction = torch.argmax(output, dim=1).item()

                    # Mover el vehículo
                    move(prediction)
                    print(f"{moving_dots} Modelo funcionando! Predicción del modelo: {id_to_key(prediction)}", end="\r")

                elif output_size == 2: # Caso de controlador, continuo. Se usa el valor directamente
                    # Obtener la predicción del modelo
                    prediction = torch.clamp(output, min=-1.0, max=1.0).tolist()[0] # Limitar los valores entre -1.0 y 1.0                                     
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

    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Ejecutar el modelo de conducción autónoma.")
    parser.add_argument("--model_path", type=str, default=fr"C:\Users\PC\Documents\GitHub\TrabajoMemoria\trained_models\{MODEL_NAME}", help="Ruta del modelo entrenado.")
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

