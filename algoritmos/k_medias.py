import numpy as np

def k_medias(imagen):
    num_clusters = 3
    max_iter = 4
   # Convertir la imagen en una matriz NumPy 1D
    datos = imagen.ravel().reshape(-1, 1)
    
    # Inicializar centroides aleatorios
    centroides = datos[np.random.choice(datos.shape[0], num_clusters, replace=False)]
    
    # Iterar para ajustar los centroides
    for _ in range(max_iter):
        # Expandir centroides para que tengan la misma forma que datos
        centroides_expandidos = centroides.reshape(1, -1)
        
        # Calcular la distancia de cada punto a los centroides
        distancias = np.abs(datos - centroides_expandidos)
        
        # Asignar puntos a los clusters según la distancia mínima
        clusters = np.argmin(distancias, axis=1)
        
        # Actualizar centroides como la media de los puntos en cada cluster
        nuevos_centroides = np.array([datos[clusters == i].mean() for i in range(num_clusters)])
        
        # Comprobar convergencia
        if np.allclose(centroides, nuevos_centroides):
            break
        
        centroides = nuevos_centroides
    
    # Reorganizar los píxeles en la forma original de la imagen
    segmentada = clusters.reshape(imagen.shape)
    
    return segmentada