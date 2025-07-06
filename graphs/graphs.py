import matplotlib.pyplot as plt
import pandas as pd

# Cambia el nombre del archivo CSV por el tuyo
csv_file = 'graphs/HIGH_supervised_vabn_stack_1s_128p_C_4img_4act_20hz_cut.csv'  # Ejemplo: 'train_episode_reward.csv'

df = pd.read_csv(csv_file)

# Asume que el CSV tiene columnas 'step' y 'value'. Cambia si tus columnas tienen otros nombres.
x = df.iloc[:, 1]
y = df.iloc[:, 2]

plt.figure(figsize=(5, 3))
plt.plot(x, y)

# Líneas punteadas horizontales
nivel_cerrada = 0.016  # Cambia por el valor relevante
nivel_bifurcacion = 0.037  # Cambia por el valor relevante
plt.axhline(nivel_cerrada, color='r', linestyle='--', label='Curva cerrada')
plt.axhline(nivel_bifurcacion, color='g', linestyle='--', label='Bifurcación')

plt.xlabel('Episodio')
plt.ylabel('Porcentaje de progreso')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
