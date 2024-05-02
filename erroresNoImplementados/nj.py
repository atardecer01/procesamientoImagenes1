import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def while_stripe_normalize(image):
    # Calcular el histograma de la imagen
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0,256))

    # Encontrar el último pico en el histograma
    last_peak_value = np.max(hist)
    # Normalizar la imagen dividiendo por el valor del último pico
    normalized_image = image / last_peak_value

    return normalized_image

# Cargar la imagen .nii
imagen_nii = nib.load("./imagen.nii")
imagen_data = imagen_nii.get_fdata()

# Normalizar la imagen utilizando el algoritmo "While Stripe"
imagen_normalizada = while_stripe_normalize(imagen_data)

# Visualizar la imagen original y la imagen normalizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen_data[:, :, imagen_data.shape[0] // 2], cmap='gray')  # Visualizar un corte axial de la imagen original
plt.title('Imagen original')
plt.subplot(1, 2, 2)
plt.imshow(imagen_normalizada[:, :, imagen_normalizada.shape[0] // 2], cmap='gray')  # Visualizar un corte axial de la imagen normalizada
plt.title('Imagen normalizada')
plt.show()
