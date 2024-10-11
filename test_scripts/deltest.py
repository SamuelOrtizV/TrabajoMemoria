import os
import glob



data_path = "./datasets/raw"

def delete_last_images(data_path, last_img_id, num_files):
    """
    Elimina las últimas imágenes capturadas.

    :param data_path: Ruta donde se guardarán las imágenes.
    :param last_img_id: Número de la última imagen guardada.
    :param num_files: Número de archivos a eliminar.
    """

    deleted_files = 0
    i = 0
    
    while deleted_files < num_files and last_img_id - i >= 0:
        pattern = os.path.join(data_path, f"{last_img_id-i}_*.jpeg")
        files = glob.glob(pattern)
        
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files += 1
                if deleted_files >= num_files:
                    break
        i += 1


def get_last_image_number(save_path):
    """
    Obtiene el número de la última imagen guardada en una carpeta.

    :param save_path: Ruta donde se guardarán las imágenes.
    :return: Número de la última imagen guardada y la cantidad de imágenes en la carpeta.
    """
    files = os.listdir(save_path)
    files = [int(f.split("_")[0]) for f in files if f.endswith(".jpeg")]

    dataset_size = len(files)

    if dataset_size == 0:
        return 0, 0

    files.sort()

    next_img_number = max(files) + 1

    return next_img_number, dataset_size

num_imgs_to_delete = 100

img_id, dataset_size = get_last_image_number(data_path)

delete_last_images(data_path, img_id, num_imgs_to_delete)

img_id, _ = get_last_image_number(data_path)

print(f"Deleted {num_imgs_to_delete} images from {data_path}, last image is {img_id-1}.")
