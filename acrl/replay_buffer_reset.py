from tmrl.util import load, dump
import tmrl.config.config_constants as cfg
import random
import matplotlib.pyplot as plt

checkpoint_path = cfg.CHECKPOINT_PATH

checkpoint = load(checkpoint_path)

replay_memory = checkpoint.memory

try:
    checkpoint.memory = []
    dump(checkpoint, checkpoint_path)
    print("Replay Buffer vaciado correctamente\n")
except Exception as e:
    print(f"No se pudo limpiar el Replay Buffer\n {e}")

""" try:
    while True:
        # Seleccionar una imagen aleatoria del buffer
        if len(replay_memory.data[3]) > 0:  # Asegúrate de que haya imágenes en el buffer
            random_index = random.randint(0, len(replay_memory.data[3]) - 1)
            random_image = replay_memory.data[3][random_index]  # Selecciona la imagen

            # Visualizar la imagen
            plt.imshow(random_image)
            plt.title(f"Imagen aleatoria del buffer (índice {random_index})")
            plt.axis('off')  # Opcional: ocultar los ejes
            plt.show()
        else:
            print("No hay imágenes en el buffer para mostrar.")
except KeyboardInterrupt:
    print("Interrumpido por el usuario. Saliendo...") """