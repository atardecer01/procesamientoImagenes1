'''import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def histogram_matching(source_hist, target_hist):
    """
    Función para realizar el pareo de histogramas.
    
    Args:
    source_hist: Histograma de la imagen de origen.
    target_hist: Histograma de la imagen objetivo.
    
    Returns:
    matched_hist: Histograma resultante después del pareo.
    """
    source_cdf = np.cumsum(source_hist) / np.sum(source_hist)
    target_cdf = np.cumsum(target_hist) / np.sum(target_hist)
    
    # Crear un mapeo de píxeles
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        idx = np.argmin(np.abs(source_cdf[i] - target_cdf))
        mapping[i] = idx
    
    matched_hist = np.zeros_like(source_hist)
    for i in range(256):
        matched_hist[mapping[i]] += source_hist[i]
    
    return matched_hist, mapping

# Cargar imágenes .nii
source_img = nib.load("./imagen.nii")

target_img = nib.load("./sub-03_T1w.nii")

# Obtener datos de las imágenes
source_data = source_img.get_fdata()
target_data = target_img.get_fdata()

# Calcular los histogramas de las imágenes
source_hist, _ = np.histogram(source_data.flatten(), bins=256, range=(0, 255))
target_hist, _ = np.histogram(target_data.flatten(), bins=256, range=(0, 255))

# Normalizar los histogramas
source_hist = source_hist / np.sum(source_hist)
target_hist = target_hist / np.sum(target_hist)

# Realizar el pareo de histogramas
matched_hist, mapping = histogram_matching(source_hist, target_hist)

# Aplicar la transformación al conjunto de datos de la imagen de origen
matched_data = mapping[source_data.astype(int)]

# Mostrar las imágenes y sus histogramas
plt.figure(figsize=(15, 6))

# Mostrar la imagen de origen
plt.subplot(2, 3, 1)
plt.imshow(source_data[:, :, source_data.shape[2] // 2], cmap='gray')
plt.title('Imagen de origen')

# Mostrar el histograma de la imagen de origen
plt.subplot(2, 3, 4)
plt.plot(source_hist)
plt.title('Histograma de la imagen de origen')

# Mostrar la imagen objetivo
plt.subplot(2, 3, 2)
plt.imshow(target_data[:, :, target_data.shape[2] // 2], cmap='gray')
plt.title('Imagen objetivo')

# Mostrar el histograma de la imagen objetivo
plt.subplot(2, 3, 5)
plt.plot(target_hist)
plt.title('Histograma de la imagen objetivo')

# Mostrar el histograma resultante después del pareo
plt.subplot(2, 3, 3)
plt.plot(matched_hist)
plt.title('Histograma resultante después del pareo')

# Mostrar la imagen resultante
plt.subplot(2, 3, 6)
plt.imshow(matched_data[:, :, matched_data.shape[2] // 2], cmap='gray')
plt.title('Imagen resultante después del pareo')

plt.tight_layout()
plt.show()
'''
'''
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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

    # Crear una nueva imagen con los datos de intensidad ajustados
    matched_img = nib.Nifti1Image(matched_img_data, img_data.affine, img_data.header)

    return matched_img

def plot_histogram(image, title):
    data = image.get_fdata()
    plt.hist(data.flatten(), bins=256, range=(0,255), density=True, color='b', alpha=0.6)
    plt.title(title)
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.show()

# Cargar imágenes
image_path = "./imagen.nii"
reference_path = "./sub-03_T1w.nii"

img = nib.load(image_path)
ref = nib.load(reference_path)

# Mostrar histogramas antes del pareo
plot_histogram(img, 'Histograma de la imagen original')
plot_histogram(ref, 'Histograma de la imagen de referencia')
img_data = img.get_fdata()
ref_data = ref.get_fdata()
# Aplicar pareo de histograma
matched_img = histogram_matching(img_data, ref_data)

# Mostrar histograma después del pareo
plot_histogram(matched_img, 'Histograma de la imagen con pareo')

# Visualizar imágenes (opcional)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img.get_fdata()[:, :, img.shape[2]//2], cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(matched_img.get_fdata()[:, :, matched_img.shape[2]//2], cmap='gray')
plt.title('Imagen con Pareo de Histograma')
plt.axis('off')
plt.show()
'''


import numpy as np
import matplotlib.pyplot as plt
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

    return matched_img_data

def plot_histogram(hist, title):
    plt.plot(hist)
    plt.title(title)
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.show()

# Cargar imágenes
source_img = nib.load("./imagen.nii")
target_img = nib.load("./sub-03_T1w.nii")

# Obtener datos de las imágenes
source_data = source_img.get_fdata()
target_data = target_img.get_fdata()

# Calcular los histogramas de las imágenes
source_hist, _ = np.histogram(source_data.flatten(), bins=256, range=(0, 255))
target_hist, _ = np.histogram(target_data.flatten(), bins=256, range=(0, 255))

# Normalizar los histogramas
source_hist = source_hist / np.sum(source_hist)
target_hist = target_hist / np.sum(target_hist)

# Realizar el pareo de histogramas
matched_data = histogram_matching(source_data, target_data)

# Mostrar las imágenes y sus histogramas
plt.figure(figsize=(15, 6))

# Mostrar la imagen de origen
plt.subplot(2, 3, 1)
plt.imshow(source_data[:, :, source_data.shape[2] // 2], cmap='gray')
plt.title('Imagen de origen')

# Mostrar el histograma de la imagen de origen
plt.subplot(2, 3, 4)
plot_histogram(source_hist, 'Histograma de la imagen de origen')

# Mostrar la imagen objetivo
plt.subplot(2, 3, 2)
plt.imshow(target_data[:, :, target_data.shape[2] // 2], cmap='gray')
plt.title('Imagen objetivo')

# Mostrar el histograma de la imagen objetivo
plt.subplot(2, 3, 5)
plot_histogram(target_hist, 'Histograma de la imagen objetivo')

# Mostrar el histograma resultante después del pareo
plt.subplot(2, 3, 3)
plot_histogram(matched_data.flatten(), 'Histograma resultante después del pareo')

# Mostrar la imagen resultante
plt.subplot(2, 3, 6)
plt.imshow(matched_data[:, :, matched_data.shape[2] // 2], cmap='gray')
plt.title('Imagen resultante después del pareo')

plt.tight_layout()
plt.show()
