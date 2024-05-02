import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def remove_stripes(image, stripe_width=5, num_iterations=10):
    # Crear una copia de la imagen original
    corrected_image = np.copy(image)
    
    # Iterar para eliminar las franjas
    for _ in range(num_iterations):
        # Calcular la media de intensidad de cada columna de píxeles
        column_means = np.mean(corrected_image, axis=(0, 1))  # Calcular la media a lo largo de los ejes x e y
        
        # Aplicar un filtro de suavizado (por ejemplo, un filtro de media) a las medias de las columnas
        smoothed_means = np.convolve(column_means, np.ones(stripe_width)/stripe_width, mode='same')
        
        # Restar las medias suavizadas de las intensidades de las columnas
        corrected_image -= smoothed_means[np.newaxis, np.newaxis, :]
    
    return corrected_image


# Cargar la imagen .nii
imagen_nii = nib.load("./imagen.nii")
imagen_data = imagen_nii.get_fdata()

# Aplicar el algoritmo "Stripe" para eliminar las franjas
imagen_corregida = remove_stripes(imagen_data)

# Visualizar la imagen original y la imagen corregida
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen_data[:, :, imagen_data.shape[2] // 2], cmap='gray')  # Visualizar un corte axial de la imagen original
plt.title('Imagen original')
plt.subplot(1, 2, 2)
plt.imshow(imagen_corregida[:, :, imagen_data.shape[2] // 2], cmap='gray')  # Visualizar un corte axial de la imagen corregida
plt.title('Imagen corregida')
plt.show()
