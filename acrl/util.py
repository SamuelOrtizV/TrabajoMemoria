import tkinter as tk  
from PIL import Image, ImageTk

class ImageVisualizer:
    def __init__(self, title="Visualización en tiempo real", position="top-right"):
        """
        Inicializa el visualizador de imágenes con Tkinter.

        Args:
            title (str): Título de la ventana.
            position (str): Posición de la ventana en la pantalla ("top-right", "top-left", etc.).
        """
        self.root = tk.Tk()
        self.root.title(title)

        # Configurar la posición de la ventana
        self.set_window_position(position)

        # Crear un widget de etiqueta para mostrar la imagen
        self.label = tk.Label(self.root)
        self.label.pack()

    def set_window_position(self, position):
        """
        Configura la posición de la ventana en la pantalla.

        Args:
            position (str): Posición de la ventana ("top-right", "top-left", etc.).
        """
        # Obtener el tamaño de la pantalla
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calcular la posición
        if position == "top-right":
            x_offset = screen_width - 300  # Dejar un margen pequeño
            y_offset = 0
        elif position == "top-left":
            x_offset = 0
            y_offset = 0
        elif position == "bottom-right":
            x_offset = screen_width - 300
            y_offset = screen_height - 300
        elif position == "bottom-left":
            x_offset = 0
            y_offset = screen_height - 300
        else:  # Centro por defecto
            x_offset = screen_width // 2
            y_offset = screen_height // 2

        # Configurar la geometría de la ventana (sin ajustar dimensiones manualmente)
        self.root.geometry(f"+{x_offset}+{y_offset}")

    def update_image(self, img):
        """
        Actualiza la imagen mostrada en la ventana.

        Args:
            img: La imagen a mostrar (numpy array).
        """
        # Convertir la imagen a un formato compatible con Tkinter
        if len(img.shape) == 2:  # Escala de grises
            img = Image.fromarray(img)
        else:  # Color RGB
            img = Image.fromarray(img, 'RGB')

        # Convertir la imagen a un objeto PhotoImage
        img_tk = ImageTk.PhotoImage(img)

        # Actualizar la imagen en el widget de etiqueta
        self.label.config(image=img_tk)
        self.label.image = img_tk

        # Actualizar la ventana
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        """
        Cierra la ventana de visualización.
        """
        self.root.destroy()