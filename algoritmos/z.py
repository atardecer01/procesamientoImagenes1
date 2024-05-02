'''import nibabel as nib

import matplotlib.pyplot as plt

img = nib.load("./imagen.nii")

img = img.get_fdata()'''
import numpy as np
def zCore(img):
    '''meanV = img[img > 10].mean()
    stdV = img[img > 10].std()

    imgZcore = (img - meanV) / stdV
    return imgZcore'''
    # Calcular la media y la desviación estándar solo en los píxeles con valores mayores que 10
    masked_img = img[img > 10]
    meanV = masked_img.mean()
    stdV = masked_img.std()

    # Aplicar la corrección z-score solo a los píxeles con valores mayores que 10
    imgZcore = np.zeros_like(img, dtype=np.float64)
    imgZcore[img > 10] = (img[img > 10] - meanV) / stdV

    # Mantener los valores de los píxeles que no cumplan la condición original
    imgZcore[img <= 10] = img[img <= 10]

    return imgZcore



'''plt.hist(imgZcore[img > 10], 100)
plt.show()'''