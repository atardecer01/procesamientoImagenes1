import numpy as np


def umbralizacion(imagen):
    umbral = np.mean(imagen)
    # Crear una copia de la imagen original
    imagen_umbralizada = np.copy(imagen)
    # Aplicar umbralización
    imagen_umbralizada[imagen_umbralizada <= umbral] = 0
    imagen_umbralizada[imagen_umbralizada > umbral] = 255
    return imagen_umbralizada
