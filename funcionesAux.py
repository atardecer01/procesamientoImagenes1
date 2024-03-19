import nibabel as nib
import numpy as np

# Función para guardar la imagen umbralizada en un archivo .nii
def guardar_imagen_umbralizada(imagen_umbralizada, ruta_guardado):
    # Crear un objeto Nibabel
    imagen_nii_umbralizada = nib.Nifti1Image(imagen_umbralizada, np.eye(4))  # Usamos una matriz identidad para la información de la imagen
    
    # Guardar la imagen en un archivo .nii
    nib.save(imagen_nii_umbralizada, ruta_guardado)