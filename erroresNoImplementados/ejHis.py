import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Cargar la imagen .nii
nii_img = nib.load('./imagen.nii')
img_data = nii_img.get_fdata()

# Convertir la imagen a escala de grises
if img_data.shape[-1] > 1:
    # Si la imagen tiene múltiples canales, elige uno (por ejemplo, el canal medio)
    img_data = img_data[..., img_data.shape[-1] // 2]

# Ecualizar el histograma
img_data_eq = cv2.equalizeHist(np.uint8(img_data))

# Mostrar la imagen ecualizada
plt.imshow(img_data_eq)
plt.title('Imagen Ecualizada')
plt.axis('off')  # Ocultar ejes
plt.show()
