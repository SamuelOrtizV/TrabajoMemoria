import matplotlib.pyplot as plt
import pandas as pd

# Cambia el nombre del archivo CSV por el tuyo

csv_file = 'ep_prog'  # Ejemplo: 'train_episode_reward.csv'

df = pd.read_csv('graphs/'+csv_file+'.csv')

# Asume que el CSV tiene columnas 'step' y 'value'. Cambia si tus columnas tienen otros nombres.
x = df.iloc[:, 1]
y = df.iloc[:, 2]

plt.figure(figsize=(5, 3))
plt.plot(x, y)

# Líneas punteadas horizontales
nivel_cerrada = 1.0  # Cambia por el valor relevante
plt.axhline(nivel_cerrada, color='r', linestyle='--', label='Línea de meta')

plt.xlabel('Episodio')
plt.ylabel('Porcenje de progreso')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Guardar como PDF vectorial para insertar en LaTeX (vectorizado)
plt.savefig('graphs/'+csv_file+'.pdf', format='pdf', bbox_inches='tight')
plt.show()
