import numpy as np
import nibabel as nib


# Reescalado de la imagen
def linear_rescale(image):
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val)
    return scaled_image

'''imagen_reescalada = linear_rescale_to_unit_range(imagen_data)

# Selecciona el corte axial
corte_axial = imagen_reescalada[:, :, imagen_reescalada.shape[2] // 2]

# Visualización del corte axial
plt.imshow(corte_axial, cmap='gray')
plt.colorbar()
plt.show()'''

#1:21:32 - explica este tema