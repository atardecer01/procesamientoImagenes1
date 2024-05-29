import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

def calcularWeights(image, beta=0.1):
    height, width = image.shape
    num_pixels = height * width
    indices = np.arange(num_pixels).reshape(height, width)
    W = np.zeros((num_pixels, num_pixels))

    sigma = np.max([
        np.max(np.abs(image[i, j] - image[i - 1, j])) if i > 0 else 0,
        np.max(np.abs(image[i, j] - image[i + 1, j])) if i < height - 1 else 0,
        np.max(np.abs(image[i, j] - image[i, j - 1])) if j > 0 else 0,
        np.max(np.abs(image[i, j] - image[i, j + 1])) if j < width - 1 else 0
    ])

    sigma = sigma if sigma != 0 else 1

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

    return W

def segmentacionImg(image, seeds, labels, xB, xF, beta):
    height, width = image.shape
    num_pixels = height * width
    indices = np.arange(num_pixels).reshape(height, width)

    W = calcularWeights(image, beta)
    D = np.sum(W, axis=1)
    L = sp.diags(D) - W

    L2 = L.dot(L)

    I_s = sp.lil_matrix((num_pixels, num_pixels))

    b = np.zeros(num_pixels)
    for (i, j), label in zip(seeds, labels):
        idx = indices[i, j]
        I_s[idx, idx] = 1
        b[idx] = xB if label == 'B' else xF

    A = I_s + L2

    A = sp.csr_matrix(A)
    solve = factorized(A)

    x = solve(b)

    segmented_image = x.reshape((height, width))
    return segmented_image
