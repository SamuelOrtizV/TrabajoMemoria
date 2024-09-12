import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image


# Definición del dataset personalizado
class RacingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)]) # Obtener las rutas de las imágenes en el directorio ordenadas alfabéticamente por nombre 
    def __len__(self):
        return len(self.image_paths) #- self.seq_len + 1
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = int(os.path.basename(image_path).split('_')[1].split(".")[0])  # Asume que la etiqueta está en el nombre despues del id y antes de la extensión

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label
    
""" class RacingDataset(Dataset):
    def __init__(self, data_dir, seq_len, transform=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.data = self.load_data()  # Cargar tus datos aquí
        self.labels = self.load_labels()  # Cargar tus etiquetas aquí

    def load_data(self):
        # Implementa la lógica para cargar tus datos
        pass

    def load_labels(self):
        # Implementa la lógica para cargar tus etiquetas
        pass

    def __len__(self):
        # El tamaño del dataset es el número de imágenes menos el tamaño de la secuencia más 1
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        # Crear una secuencia de imágenes
        images = self.data[idx:idx + self.seq_len]
        label = self.labels[idx + self.seq_len - 1]  # La etiqueta es la de la última imagen en la secuencia
        if self.transform:
            images = [self.transform(image) for image in images]
        images = torch.stack(images)  # Convertir la lista de imágenes en un tensor
        return images, label """

# Definición del modelo    
class SimpleRNN(nn.Module):
    """
    Modelo de red neuronal recurrente simple que utiliza una CNN preentrenada para extraer características de las imágenes
    y una capa RNN para procesar la secuencia de características.

    Args:
        pretrained_cnn (torch.nn.Module): Modelo de CNN preentrenado.
        hidden_size (int): Tamaño del estado oculto de la RNN.
        output_size (int): Tamaño de la salida del modelo.
        input_size (tuple, optional): Tamaño de la entrada de las imágenes (canales, altura, ancho). Máximo tamaño es 384x384.
        num_layers (int, optional): Número de capas de la RNN. 
        dropout (float, optional): Probabilidad de dropout.
        bias (bool, optional): Si se incluye sesgo en las capas lineales.
    """
    def __init__(self, pretrained_cnn, hidden_size, output_size, input_size=(3, 224, 224), num_layers=1, dropout=0, bias=True):
        super(SimpleRNN, self).__init__()
        
        # Usar el modelo preentrenado
        self.cnn = pretrained_cnn
        self.cnn.classifier = nn.Identity()  # Quitar la última capa de clasificación
        #self.cnn.eval()  # Poner el modelo en modo evaluación
        
        # Desactivar el gradiente para las capas convolucionales preentrenadas, de esta forma no se actualizarán los pesos durante el entrenamiento
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Obtener el tamaño de la salida de la CNN
        self.input_size = input_size
        conv_output_size = self._get_conv_output_size()
        
        # Definir la capa RNN y la capa totalmente conectada
        self.rnn = nn.RNN(conv_output_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bias=bias)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def _get_conv_output_size(self):
        # Crear una entrada dummy para pasarla a través de la CNN
        dummy_input = torch.zeros(1, *self.input_size)  # Ajustar según la entrada esperada por la CNN segun los parametros ()
        output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        conv_out = []
        for t in range(seq_len):
            # Pasar cada imagen a través de la CNN preentrenada
            cnn_output = self.cnn(x[:, t, :, :, :])
            conv_out.append(cnn_output.view(batch_size, -1))
        conv_out = torch.stack(conv_out, dim=1)
        rnn_out, _ = self.rnn(conv_out)
        final_out = self.fc(rnn_out[:, -1, :])
        return final_out


