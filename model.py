import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image

class RacingDataset(Dataset):
    def __init__(self, data_dir, seq_len, transform=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        #self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
        self.data = self.load_data()  # Cargar tus datos aquí
        self.labels = self.load_labels()  # Cargar tus etiquetas aquí
        self.data_dimension = self.get_data_dimension()

    # CREO QUE AL CARGAR LOS STRINGS VA A DESORDENAR LOS DATOS, PROBAR IMPRIMIENDO LOS NOMBRES DE LOS ARCHIVOS

    def load_data(self):
        data = []
        for file in os.listdir(self.data_dir):
            image = Image.open(os.path.join(self.data_dir, file)).convert('RGB')
            data.append(image)
        return data        

    def load_labels(self):
        labels = []
        for file in os.listdir(self.data_dir):
            label = int(file.split('_')[1].split(".")[0])
            labels.append(label)
        return labels
    
    def get_data_dimension(self):
        image = self.data[0]
        return image.size

    def __len__(self):
        # El tamaño del dataset es el número de imágenes menos el tamaño de la secuencia más 1
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        # Crear una secuencia de imágenes
        
        if idx >= self.seq_len - 1: 
            images_seq = self.data[idx - self.seq_len : idx]     
        else:
            print("ALOOO")
            images_seq = []
            # Comenzar desde idx - self.seq_len + 1 y terminar en idx
            for i in range(self.seq_len):
                current_idx = idx - self.seq_len + 1 + i
                if self.data[current_idx] is not None:
                    images_seq.append(self.data[current_idx])
                else:
                    images_seq.append(Image.new('RGB', self.data_dimension, (0, 0, 0)))            

        label = self.labels[idx]

        if self.transform:
            images_seq = [self.transform(image) for image in images_seq]

        # Convertir la lista de imágenes a un tensor de tamaño (seq_len, Channels, Height, Width)

        if len(images_seq) > 0:
            images_seq = torch.stack(images_seq)
        else:
            raise RuntimeError("La lista de imágenes está vacía, no se puede apilar.")

        return images_seq, label
    
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