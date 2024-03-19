import numpy as np

def isodata(image):
  # Convertir la imagen a un array de NumPy.
  image_array = np.array(image)

  # Calcular la media y la desviación estándar de la intensidad de los píxeles.
  mean = np.mean(image_array)
  std = np.std(image_array)

  # Establecer un umbral inicial.
  old_threshold = float('inf')
  threshold = mean + std

  # Agrupar los píxeles en dos grupos según el umbral.
  groups = np.zeros_like(image_array)
  groups[image_array > threshold] = 1

  # Calcular las medias de los dos grupos.
  group_means = np.zeros(2)
  for i in range(2):
    group_means[i] = np.mean(image_array[groups == i])

  # Actualizar el umbral.
  threshold = np.mean(group_means)

  # Repetir los pasos 4 a 7 hasta que el umbral converja.
  while abs(threshold - old_threshold) > 1e-6:
    old_threshold = threshold

    groups = np.zeros_like(image_array)
    groups[image_array > threshold] = 1

    for i in range(2):
      group_means[i] = np.mean(image_array[groups == i])

    threshold = np.mean(group_means)

  # Devolver la imagen segmentada.
  return groups