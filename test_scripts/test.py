""" name = "54_-0.33 1.00.jpeg"

id = name.split("_")[0]
label = name.split("_")[1].split(".")[:-1]
extension = name.split(".")[-1]

print("ID:", id,
      "Label:", label,
      "Extension:" , extension) """

# Verificación adicional
filename = "54_-0.33 1.00.jpeg"
parts = filename.split('_')
label_with_extension = parts[1]
label = label_with_extension.rsplit('.', 1)[0]
print(label)  # Output: -0.33 1.00

giro = label.split(" ")[0]
acelerador = label.split(" ")[1]

print(giro, "\n", acelerador)  # Output: -0.33 1.00