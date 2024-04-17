import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagen_y_histograma(imagen):
    # Cargar la imagen NIfTI
    img_data = imagen.get_fdata()

    # Mostrar la imagen
    plt.figure(figsize=(10, 6))
    plt.imshow(img_data[:, :, img_data.shape[2] // 2], cmap='gray')
    plt.title('Imagen')
    plt.colorbar()
    plt.show()

    # Calcular el histograma de los datos de la imagen

    hist, bins = np.histogram(img_data[img_data > 10], bins=400)

    uno = img_data[img_data > 10]
    # Mostrar el histograma
    plt.figure()
    plt.hist( uno, bins=150, color='c')
    plt.title('Histograma de imagen de referencia')
    plt.xlabel('Intensidad de Pixel')
    plt.ylabel('Frecuencia')
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar la imagen NIfTI
    imagen = nib.load('./imgH3.nii')

    # Mostrar la imagen y su histograma
    mostrar_imagen_y_histograma(imagen)
