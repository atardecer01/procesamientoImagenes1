import numpy as np
import nibabel as nib
import spicy


def Curvatura(data):

    kernel_x = np.array(
        [-1, 1]
    ) / 2

    Dx = spicy.ndimage.convolve(data,kernel_x.reshape((2,1,1)))
    Dxx = spicy.ndimage.convolve(Dx,kernel_x.reshape((2,1,1)))
    Dy = spicy.ndimage.convolve(data,kernel_x.reshape((1,2,1)))
    Dyy = spicy.ndimage.convolve(Dy,kernel_x.reshape((1,2,1)))
    Dz = spicy.ndimage.convolve(data,kernel_x.reshape((1,1,2)))
    Dzz = spicy.ndimage.convolve(Dz,kernel_x.reshape((1,1,2)))

    img_filt_data = np.sqrt(Dxx ** 2 + Dyy ** 2 + Dzz ** 2)

    img_filt_nifti = nib.Nifti1Image(img_filt_data,  np.eye(4))

    # Guardar la imagen .nii transformada
    nib.save(img_filt_nifti, 'imagen_transformadaCurva.nii')

    return img_filt_data