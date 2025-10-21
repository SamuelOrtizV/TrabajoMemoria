import tkinter as tk  
from PIL import Image, ImageTk


class ImageVisualizer:
    def __init__(self, title="Real-time visualization", position="top-right"):
        """Initialize the Tkinter image viewer window.

        Args:
            title: window title
            position: window position on the screen ("top-right", "top-left", etc.)
        """
        self.root = tk.Tk()
        self.root.title(title)

        # Set window position
        self.set_window_position(position)

        # Label widget to display the image
        self.label = tk.Label(self.root)
        self.label.pack()

    def set_window_position(self, position):
        """Set the window position on screen.

        Args:
            position: one of "top-right", "top-left", "bottom-right", "bottom-left", or center by default
        """
        # Screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Position calculation
        if position == "top-right":
            x_offset = screen_width - 300  # small side margin
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
        else:  # default center
            x_offset = screen_width // 2
            y_offset = screen_height // 2

        # Apply geometry
        self.root.geometry(f"+{x_offset}+{y_offset}")

    def update_image(self, img):
        """Update the image displayed in the window.

        Args:
            img: numpy array image to display (grayscale or RGB)
        """
        # Convert to Tk-compatible image
        if len(img.shape) == 2:  # grayscale
            img = Image.fromarray(img)
        else:  # RGB
            img = Image.fromarray(img, 'RGB')

        # To PhotoImage
        img_tk = ImageTk.PhotoImage(img)

        # Update label content
        self.label.config(image=img_tk)
        self.label.image = img_tk

        # Refresh window
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        """Close the visualization window."""
        self.root.destroy()