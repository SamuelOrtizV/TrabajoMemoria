import requests

# URL de la API local de SimHub
url = "http://localhost:8080/api/v1/current"

# Realiza la solicitud GET para obtener los datos actuales
response = requests.get(url)
data = response.json()

# Extrae algunos datos importantes
speed = data['CarState']['SpeedKmh']
lap_time = data['Timing']['LastLapTime']
out_of_track = data['Physics']['IsOffTrack']

print(f"Velocidad: {speed} km/h")
print(f"Último tiempo de vuelta: {lap_time}")
print(f"Está fuera de pista: {out_of_track}")
