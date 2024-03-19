import numpy as np
from queue import Queue


def crecimiento_regiones(imagen):
    umbral = np.mean(imagen)
    seed = (88,96,96)
    # Obtener las dimensiones de la imagen
    dim_x, dim_y, dim_z = imagen.shape

    # Crear una matriz para marcar los píxeles procesados
    labeled = np.zeros_like(imagen)

    # Crear una cola para almacenar los píxeles a procesar
    que = Queue()
    # Agregar la semilla a la cola
    que.put(seed)

    # Definir el valor de la semilla
    seed_value = imagen[seed]

    # Mientras la cola no esté vacía
    while not que.empty():
        # Obtener las coordenadas del píxel de la cola
        current_pixel = que.get()
        x, y, z = current_pixel

        # Verificar si el píxel ya ha sido procesado
        if labeled[x, y, z] == 1:
            continue

        # Marcar el píxel como procesado
        labeled[x, y, z] = 1

        # Agregar los vecinos que coincidan con el valor de la semilla dentro del umbral
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (0 <= x+i < dim_x and 0 <= y+j < dim_y and 0 <= z+k < dim_z):
                        diff = np.abs(imagen[x+i, y+j, z+k] - seed_value)
                        if diff <= umbral:
                            que.put((x+i, y+j, z+k))

    # Crear una imagen binaria con las regiones que mejor coincidan con la semilla
    imagen_pintada = np.zeros_like(imagen)
    imagen_pintada[np.where(labeled == 1)] = imagen[np.where(labeled == 1)]

    return imagen_pintada