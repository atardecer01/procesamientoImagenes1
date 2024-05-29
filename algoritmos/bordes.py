import numpy as np
import nibabel as nib
import spicy


def bordes(data):
    # Definir el kernel
    kernel_x = np.array(
        [-1, 1]
    ) / 2

    # Aplicar la convolución a lo largo de cada eje
    img_filt_x = spicy.ndimage.convolve(data,kernel_x.reshape((2,1,1)))
    img_filt_y = spicy.ndimage.convolve(data,kernel_x.reshape((1,2,1)))
    img_filt_z = spicy.ndimage.convolve(data,kernel_x.reshape((1,1,2)))

    # Crear una nueva imagen .nii a partir de la matriz de datos transformada
    img_filt_data = np.sqrt(img_filt_x ** 2 + img_filt_y ** 2 + img_filt_z ** 2)
    img_filt_nifti = nib.Nifti1Image(img_filt_data,  np.eye(4))

    # Guardar la imagen .nii transformada
    nib.save(img_filt_nifti, 'imagen_transformada.nii')

    return img_filt_data
