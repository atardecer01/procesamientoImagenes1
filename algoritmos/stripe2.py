import numpy as np
import nibabel as nib

def while_stripe(image,v):
    img = image[image > 0]
   # Calcular el histograma de la imagen
    hist, bins = np.histogram(img.flatten(), bins=200)

    # Encontrar el primer pico más alto comenzando desde la derecha del histograma
    indice_pico = len(hist) - 1 - np.argmax(hist[::-1])

    # Obtener el valor del primer pico encontrado
    moda = bins[indice_pico]

    # Normalizar la imagen dividiendo por el valor del último pico
    normalized_image = image / moda

    # Crear un objeto NIfTI con los datos procesados
    matched_nii = nib.Nifti1Image(normalized_image, np.eye(4))  # Aquí se asume que la imagen es 3D y tiene una matriz de transformación identidad
    
    # Guardar la imagen en un archivo .nii
    nib.save(matched_nii, 'imgH1.nii')
    return normalized_image


'''# Cargar la imagen .nii
imagen_nii = nib.load("./imagen.nii")
imagen_data = imagen_nii.get_fdata()

# Normalizar la imagen utilizando el algoritmo "While Stripe"
imagen_normalizada = while_stripe_normalize(imagen_data)

# Visualizar la imagen original y la imagen normalizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen_data[:, :, imagen_data.shape[2] // 2], cmap='gray')  # Visualizar un corte axial de la imagen original
plt.title('Imagen original')
plt.subplot(1, 2, 2)
plt.imshow(imagen_normalizada[:, :, imagen_data.shape[2] // 2], cmap='gray')  # Visualizar un corte axial de la imagen normalizada
plt.title('Imagen normalizada')
plt.show()
'''