'''
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(source_data, template_data):
 # Normalizar los valores de source_data al rango de 0 a 1
    source_data_normalized = (source_data - np.min(source_data)) / (np.max(source_data) - np.min(source_data))

    # Normalizar los valores de template_data al rango de 0 a 1
    template_data_normalized = (template_data - np.min(template_data)) / (np.max(template_data) - np.min(template_data))

    # Calcular los histogramas de las imágenes normalizadas
    source_hist, _ = np.histogram(source_data_normalized.flatten(), bins=256, range=(0, 1))
    template_hist, _ = np.histogram(template_data_normalized.flatten(), bins=256, range=(0, 1))

    # Normalizar los histogramas acumulativos
    source_cdf = source_hist.cumsum() / source_hist.sum()
    template_cdf = template_hist.cumsum() / template_hist.sum()

    # Mapear los valores de intensidad usando la transformación
    lut = np.interp(source_cdf, template_cdf, range(256))
    # Redondear los valores de lut
    lut_rounded = np.round(lut).astype(np.uint8)

    # Utilizar los valores redondeados de lut como índices para acceder a los valores de intensidad
    matched_data = lut_rounded[(source_data_normalized * 255).astype(np.uint8)]
    
    # Crear un objeto NIfTI con los datos procesados
    matched_nii = nib.Nifti1Image(matched_data, np.eye(4))  # Aquí se asume que la imagen es 3D y tiene una matriz de transformación identidad
    
    # Guardar la imagen en un archivo .nii
    nib.save(matched_nii, 'imgH.nii')

    return matched_data
    # Cargar imágenes .nii
source_img = nib.load("./imagen.nii")
target_img = nib.load("./sub-03_T1w.nii")'''

'''# Cargar las imágenes NIfTI
source_image = nib.load('./imagen.nii').get_fdata()
template_image = nib.load('./sub-03_T1w.nii').get_fdata()

# Aplicar el pareo de histograma
matched_image_data = histogram_matching(source_image, template_image)

# Mostrar las imágenes resultantes después del pareo del histograma
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(source_image[:, :, source_image.shape[2] // 2], cmap='gray')
plt.title('Source Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(template_image[:, :, template_image.shape[2] // 2], cmap='gray')
plt.title('Template Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(matched_image_data[:, :, matched_image_data.shape[2] // 2], cmap='gray')
plt.title('Matched Image')
plt.axis('off')

plt.show()

import nibabel as nib
import numpy as np

def histogram_matching(input_image, reference_image):
    # Cargar las imágenes de entrada y de referencia
    input_nifti = nib.load(input_image)
    reference_nifti = nib.load(reference_image)
    
    # Obtener los datos de las imágenes
    input_data = input_nifti.get_fdata()
    reference_data = reference_nifti.get_fdata()
    
    # Calcular los histogramas de las imágenes de entrada y de referencia
    hist_img, bins_img = np.histogram(input_data.flatten(), bins=256, range=(0, 255))
    hist_ref, bins_ref = np.histogram(reference_data.flatten(), bins=256, range=(0, 255))
    
    # Calcular las funciones de distribución acumulativa (CDF) de los histogramas
    cdf_img = hist_img.cumsum()
    cdf_ref = hist_ref.cumsum()
    
    # Normalizar las CDF para asegurar que tengan el mismo rango dinámico
    cdf_img = (cdf_img / cdf_img[-1]) * 255
    cdf_ref = (cdf_ref / cdf_ref[-1]) * 255
    
    # Crear un mapeo de intensidades de píxeles basado en las CDF
    lut = np.interp(input_data.flat, bins_img[:-1], cdf_ref)
    
    # Redimensionar el mapeo a la forma de la imagen original
    lut_reshaped = lut.reshape(input_data.shape)
    
    # Aplicar el mapeo a la imagen de entrada
    matched_data = lut_reshaped.astype(input_data.dtype)
    
    # Crear un nuevo objeto NIfTI con los datos emparejados
    matched_nifti = nib.Nifti1Image(matched_data, input_nifti.affine, input_nifti.header)
    
    return matched_nifti, input_data, matched_data, hist_img, hist_ref, bins_img, bins_ref'''
'''

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def histogram_matching(image1, image2):
    # Cargar las imágenes NIfTI
    img1 = nib.load(image1)
    img2 = nib.load(image2)
    
    # Obtener los datos de las imágenes
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Calcular los histogramas de ambas imágenes
    hist1, bins1 = np.histogram(data1.flatten(), bins=256, range=(0, 255))
    hist2, bins2 = np.histogram(data2.flatten(), bins=256, range=(0, 255))

    # Realizar el pareo de histogramas (en este caso, simplemente mostramos los histogramas)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist1, color='blue')
    plt.title('Histograma de Imagen 1')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 2, 2)
    plt.plot(hist2, color='red')
    plt.title('Histograma de Imagen 2')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')

    plt.show()

# Rutas de las imágenes NIfTI
imagen1 = nib.load("./imagen.nii")
imagen2 = nib.load("./sub-03_T1w.nii")

# Llamar a la función para realizar el pareo de histogramas
histogram_matching(imagen1, imagen2)'''

import numpy as np
import nibabel as nib

def histogram_matching(img_data, ref_data):
    # Obtener histogramas de las imágenes
    hist_img, bins_img = np.histogram(img_data.flatten(), bins=256, range=(0, 255))
    hist_ref, bins_ref = np.histogram(ref_data.flatten(), bins=256, range=(0, 255))

    # Calcular funciones de distribución acumulativa (CDF)
    cdf_img = hist_img.cumsum() / hist_img.sum()
    cdf_ref = hist_ref.cumsum() / hist_ref.sum()

    # Mapeo de intensidades
    mapping = np.interp(cdf_img, cdf_ref, bins_ref[:-1])

    # Aplicar mapeo a la imagen original
    matched_img_data = np.interp(img_data.flatten(), bins_img[:-1], mapping).reshape(img_data.shape)

    # Crear un objeto NIfTI con los datos procesados
    matched_nii = nib.Nifti1Image(matched_img_data, np.eye(4))  # Aquí se asume que la imagen es 3D y tiene una matriz de transformación identidad
    
    # Guardar la imagen en un archivo .nii
    nib.save(matched_nii, 'imgH3.nii')
   
    return matched_img_data