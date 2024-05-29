
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

# Mount Google Drive


import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, factorized


def calculate_weights(image, beta=0.1):
    height, width = image.shape
    num_pixels = height * width
    indices = np.arange(num_pixels).reshape(height, width)
    W = lil_matrix((num_pixels, num_pixels))

    # Primero encontrar sigma, la máxima diferencia de intensidad entre vecinos
    sigma = 0.0
    for i in range(height):
        for j in range(width):
            if i > 0:  # Arriba
                sigma = max(sigma, np.abs(float(image[i, j]) - float(image[i - 1, j])))
            if i < height - 1:  # Abajo
                sigma = max(sigma, np.abs(float(image[i, j]) - float(image[i + 1, j])))
            if j > 0:  # Izquierda
                sigma = max(sigma, np.abs(float(image[i, j]) - float(image[i, j - 1])))
            if j < width - 1:  # Derecha
                sigma = max(sigma, np.abs(float(image[i, j]) - float(image[i, j + 1])))

    # Asegurarse de que sigma no sea cero para evitar división por cero
    if sigma == 0:
        sigma = 1.0

    # Calcular los pesos con el sigma encontrado
    for i in range(height):
        for j in range(width):
            index = indices[i, j]
            if i > 0:  # Arriba
                W[index, indices[i - 1, j]] = np.exp(-beta * (np.sqrt(np.abs(image[i, j] - image[i - 1, j])**2) / sigma))
            if i < height - 1:  # Abajo
                W[index, indices[i + 1, j]] = np.exp(-beta * (np.sqrt(np.abs(image[i, j] - image[i + 1, j])**2) / sigma))
            if j > 0:  # Izquierda
                W[index, indices[i, j - 1]] = np.exp(-beta * (np.sqrt(np.abs(image[i, j] - image[i, j - 1])**2) / sigma))
            if j < width - 1:  # Derecha
                W[index, indices[i, j + 1]] = np.exp(-beta * (np.sqrt(np.abs(image[i, j] - image[i, j + 1])**2) / sigma))

    return W.tocsr() 


def segment_image(image, seeds, labels, xB, xF, beta):
    height, width = image.shape
    num_pixels = height * width
    indices = np.arange(num_pixels).reshape(height, width)

    W = calculate_weights(image, beta)
    D = np.array(W.sum(axis=1)).flatten()
    print("D shape: ",D.shape)
    L = sp.diags(D) - W

    L2 = L.dot(L)

    # Preparar I_s, donde sólo los elementos de las semillas son 1
    I_s = sp.lil_matrix((num_pixels, num_pixels))

    b = np.zeros(num_pixels)
    for (i, j), label in zip(seeds, labels):
        idx = indices[i, j]
        I_s[idx, idx] = 1
        b[idx] = xB if label == 'B' else xF

    A = I_s + L2

    # Usando factorización Cholesky para resolver el sistema
    A = sp.csr_matrix(A)
    solve = factorized(A)  # Factorización Cholesky

    x = solve(b)

    segmented_image = x.reshape((height, width))
    return segmented_image

from PIL import Image
import numpy as np

def load_image_to_numpy(path):
    # Open the image file
    with Image.open(path) as img:
        # Convert the image to grayscale if needed
        img_gray = img.convert('L')  # Use 'RGB' to keep it in color

        # Convert the image to a NumPy array
        image_array = np.array(img_gray, dtype=float) 
        print("linea 95",image_array)
        return image_array

image_pathss = 'ejemplo.png'
data = load_image_to_numpy(image_pathss)
#image = np.random.rand(100, 100)
image = data

# Variables globales para almacenar las semillas
seeds = []
labels = []

# Función para manejar los clics del ratón
def onclick(event):
    ix, iy = int(event.xdata), int(event.ydata)
    seeds.append((iy, ix))  # Almacenar las coordenadas (fila, columna)
    print(f'Punto seleccionado: ({iy}, {ix})')

    # Añadir etiqueta basada en la selección
    if event.button == 1:  # Botón izquierdo del ratón
        labels.append('B')  # Fondo
    elif event.button == 3:  # Botón derecho del ratón
        labels.append('F')  # Objeto

    # Dibujar un punto en la posición seleccionada
    ax.plot(ix, iy, 'ro' if event.button == 1 else 'go')  # Rojo para fondo, verde para objeto
    fig.canvas.draw()

# Mostrar la imagen y conectar la función de clics
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Imprimir las semillas y etiquetas seleccionadas
print('Seeds:', seeds)
print('Labels:', labels)

print("linea 102",image.shape)
plt.imshow(image)
'''seeds = [
    # Fondo
    (88, 14), (76, 45),(80, 35),(59, 76), (26, 104), (20, 220), (10, 130),(5, 165),(30,0),(20,40),(239, 265),(60,0),(238,0),(100, 256),
      (130,18), (40,0),(160,22), (190,28), (0,190), (170,250), (230,200), (220,150), (220,100),(240,180), (1,160)
      ,(50,250),(220,60), (130,255), (90,5),(70, 40), (84, 25), (79, 46), (45, 89), (28, 111), (5, 177), (154, 26), (169, 33), (194, 42), 
      (209, 54), (210, 73), (205, 101), (201, 121), (197, 138), (230, 173), (210, 223), (147, 249), (70, 247), (31, 230),
    # Perro izquierdo
    (88, 15), (83, 45), (59, 80), (31, 108), (14, 134),(6, 166),(12, 194),(27, 222),(43, 237),(67, 245),(104, 250),
    (132, 250),(167, 241), (203, 221), (227, 178),(221, 161),(209, 146),(195, 138),(199, 108),(211, 60),(186, 38),
    (168, 37),(151, 28),(115, 21)
]
labels = [
    'B', 'B', 'B', 'B','B', 'B', 'B', 'B','B', 'B','B', 'B', 'B', 'B', 'B',
    'B','B', 'B', 'B', 'B','B','B', 'B','B','B', 'B',  'B', 'B','B', 'B', 'B',
    'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
    'F', 'F', 'F', 'F','F', 'F', 'F', 'F','F','F','F','F','F'
    'F', 'F', 'F', 'F','F', 'F', 'F', 'F','F','F','F',
]'''
seeds= [(86, 14), (85, 19), (84, 24), (83, 28), (83, 32), (81, 35), (81, 40), (81, 43), (80, 47), (78, 50),
        (76, 53), (75, 56), (73, 59), (70, 62), (68, 66), (67, 70), (64, 73), (60, 75), (58, 78), (55, 80),
        (53, 83), (50, 85), (45, 87), (43, 90), (47, 87), (40, 92), (39, 95), (37, 97), (35, 99), (33, 101), 
        (31, 103), (30, 105), (28, 107), (28, 110), (26, 112), (24, 114), (22, 118), (19, 121), (17, 123), 
        (16, 126), (14, 129), (13, 132), (11, 136), (10, 140), (8, 143), (8, 148), (6, 151), (6, 154), 
        (6, 158), (6, 161), (5, 164), (6, 169), (5, 174), (5, 178), (7, 182), (7, 186), (8, 189), (9, 193), 
        (11, 197), (12, 199), (13, 201), (15, 204), (15, 206), (17, 209), (19, 211), (20, 214), (22, 217), 
        (24, 219), (25, 222), (27, 223), (28, 226), (30, 228), (33, 231), (35, 233), (37, 235), (39, 237), 
        (42, 239), (44, 240), (47, 241), (50, 242), (53, 244), (56, 245), (59, 245), (62, 246), (65, 247), 
        (68, 248), (70, 248), (76, 249), (80, 250), (84, 250), (89, 250), (93, 251), (96, 251), (100, 251), 
        (104, 252), (108, 252), (112, 253), (116, 253), (122, 253), (129, 253), (137, 252), (143, 250), (150, 250), 
        (155, 249), (159, 247), (164, 246), (167, 244), (171, 244), (176, 242), (182, 240), (188, 236), (194, 232), 
        (201, 229), (206, 226), (210, 220), (215, 215), (219, 209), (223, 203), (226, 194), (230, 185), (231, 172), 
        (228, 163), (224, 159), (212, 147), (218, 150), (208, 139), (200, 137), (201, 126), (204, 119), (206, 109), 
        (209, 99), (214, 86), (217, 72), (218, 62), (211, 55), (205, 48), (195, 40), (185, 37), (174, 35), (164, 32), 
        (154, 26), (142, 24), (130, 21), (117, 18), (107, 17), (99, 14), (90, 13), (213, 77), (209, 90), (123, 20), 
        (90, 17), (86, 27), (85, 37), (82, 49), (73, 63), (66, 75), (56, 84), (46, 94), (36, 105), (26, 117), (16, 131), 
        (10, 146), (9, 156), (8, 174), (10, 183), (12, 193), (17, 204), (23, 213), (28, 218), (28, 218), (35, 228), 
        (47, 236), (55, 240), (63, 241), (71, 244), (79, 245), (89, 246), (97, 246), (103, 246), (109, 248), (114, 248), 
        (120, 248), (124, 248), (132, 247), (139, 246), (146, 245), (154, 243), (162, 241), (170, 239), (175, 237), 
        (180, 235), (185, 232), (189, 230), (195, 227), (201, 222), (205, 217), (210, 212), (214, 206), (216, 200), 
        (220, 193), (224, 187), (225, 179), (225, 171), (222, 163), (215, 153), (206, 148), (197, 141), 
        (195, 133), (197, 123), (200, 110), (201, 98), (206, 83), (208, 70), (209, 60), (199, 49), (186, 41), (174, 38), 
        (165, 35), (156, 30), (135, 25), (113, 21), (102, 19), (95, 18), (122, 23), (144, 28)]
labels= ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
         'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
         'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
         'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
         'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
         'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
         'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'F', 'F', 'F', 'F', 
         'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'B', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 
         'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 
         'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 
         'F', 'F', 'F', 'F', 'F', 'F']

xB = 130
xF = 20
beta = 1


def apply_labels(segmented_values,image, xB, xF):
    # Calcular el umbral basado en xB y xF
    threshold = (xB + xF) / 2

    # Asignar etiquetas basado en el umbral
    labels = np.where(segmented_values >= threshold, xB, image)

    return labels
final = segment_image(image, seeds, labels, xB, xF, beta)
plt.imshow(apply_labels(final,image,xB,xF))
plt.show()