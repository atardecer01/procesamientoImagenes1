import numpy as np

#histogram matching, haciendo uso de la técnica de ecualización de histograma estándar para mejora de contraste.
def histogram(image):
    # Calcular el histograma de la imagen
    hist, bins = np.histogram(image.flatten(), bins=360, range=[0,360])

    # Calcular la función de distribución acumulada (CDF)
    cdf = hist.cumsum()

    # Normalizar el CDF
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Aplicar ecualización del histograma
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Reajustar la imagen ecualizada a la forma original
    equalized_image = equalized_image.reshape(image.shape)

    return equalized_image

