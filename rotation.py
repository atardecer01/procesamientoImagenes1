import numpy as np
import nibabel as nib
from scipy.ndimage import rotate

# Cargar la imagen .nii
img_path = './imagen.nii'
img = nib.load(img_path)
data = img.get_fdata()

# Rotar la imagen 10 grados a la derecha (en el sentido de las manecillas del reloj)
rotated_data = rotate(data, angle=10, axes=(0, 1), reshape=False)

# Crear una nueva imagen .nii con los datos rotados
rotated_img = nib.Nifti1Image(rotated_data, affine=img.affine)
nib.save(rotated_img, './imagen_rotada.nii')
