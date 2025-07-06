import pandas as pd

# Configura el nombre del archivo CSV de entrada y salida
input_csv = '../HIGH_supervised_vabn_stack_1s_128p_C_4img_4act_20hz.csv'
output_csv = 'HIGH_supervised_vabn_stack_1s_128p_C_4img_4act_20hz_cut.csv'

# Episodios a eliminar (inclusive)
start_cut = 36
end_cut = 66

# Leer el CSV
# El archivo tiene columnas: Wall time,Step,Value
df = pd.read_csv(input_csv)

# Filtrar los episodios fuera del rango a eliminar
df_filtered = df[(df['Step'] < start_cut) | (df['Step'] > end_cut)].copy()

# Reindexar los steps para que sean consecutivos
df_filtered = df_filtered.reset_index(drop=True)
df_filtered['Step'] = range(len(df_filtered))

# Guardar el nuevo archivo CSV
print(f"Guardando archivo filtrado en: {output_csv}")
df_filtered.to_csv(output_csv, index=False)

print("Listo. Puedes revisar el nuevo archivo CSV.")
