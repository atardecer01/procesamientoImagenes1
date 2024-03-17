import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from queue import Queue

# Funciones de procesamiento de imágenes
def umbralizacion(imagen):
    umbral = np.mean(imagen)
    # Crear una copia de la imagen original
    imagen_umbralizada = np.copy(imagen)
    # Aplicar umbralización
    imagen_umbralizada[imagen_umbralizada <= umbral] = 0
    imagen_umbralizada[imagen_umbralizada > umbral] = 255
    return imagen_umbralizada

def isodata(image):
  # Convertir la imagen a un array de NumPy.
  image_array = np.array(image)

  # Calcular la media y la desviación estándar de la intensidad de los píxeles.
  mean = np.mean(image_array)
  std = np.std(image_array)

  # Establecer un umbral inicial.
  old_threshold = float('inf')
  threshold = mean + std

  # Agrupar los píxeles en dos grupos según el umbral.
  groups = np.zeros_like(image_array)
  groups[image_array > threshold] = 1

  # Calcular las medias de los dos grupos.
  group_means = np.zeros(2)
  for i in range(2):
    group_means[i] = np.mean(image_array[groups == i])

  # Actualizar el umbral.
  threshold = np.mean(group_means)

  # Repetir los pasos 4 a 7 hasta que el umbral converja.
  while abs(threshold - old_threshold) > 1e-6:
    old_threshold = threshold

    groups = np.zeros_like(image_array)
    groups[image_array > threshold] = 1

    for i in range(2):
      group_means[i] = np.mean(image_array[groups == i])

    threshold = np.mean(group_means)

  # Devolver la imagen segmentada.
  return groups


def crecimiento_regiones(imagen_binaria):
    seed = (50,50,50)
    # Comprobar si la imagen es binaria
    if len(imagen_binaria.shape) != 3:
        raise ValueError("La imagen debe ser tridimensional")

    # Crear una matriz para marcar los píxeles procesados
    labeled = np.zeros_like(imagen_binaria)
    dim_x, dim_y, dim_z = imagen_binaria.shape

    # Crear una cola para almacenar los píxeles a procesar
    que = Queue()
    # Agregar la semilla a la cola
    que.put(seed)

    # Definir el valor de la semilla
    seed_value = imagen_binaria[seed]

    # Mientras la cola no esté vacía
    while not que.empty():
        # Obtener las coordenadas del píxel de la cola
        current_pixel = que.get()
        x, y, z = current_pixel

        # Verificar si el píxel ya ha sido procesado
        if labeled[x, y, z] == 1:
            continue

        # Marcar el píxel como procesado
        labeled[x, y, z] = 1

        # Agregar los vecinos con el mismo valor que la semilla a la cola
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (0 <= x+i < dim_x and 0 <= y+j < dim_y and 0 <= z+k < dim_z and
                            imagen_binaria[x+i, y+j, z+k] == seed_value):
                        que.put((x+i, y+j, z+k))

    
    return labeled



# Función para guardar la imagen umbralizada en un archivo .nii
def guardar_imagen_umbralizada(imagen_umbralizada, ruta_guardado):
    # Crear un objeto Nibabel
    imagen_nii_umbralizada = nib.Nifti1Image(imagen_umbralizada, np.eye(4))  # Usamos una matriz identidad para la información de la imagen
    
    # Guardar la imagen en un archivo .nii
    nib.save(imagen_nii_umbralizada, ruta_guardado)






def k_medias(imagen):
    num_clusters = 3
    max_iter = 100
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



# Crear una ventana de tkinter
root = tk.Tk()
root.title("Visualización de imagen")

# Crear una figura de matplotlib
fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)

def mostrar_cortes():
    ruta_archivo = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")])
    if ruta_archivo:
        # Cargar la imagen
        imagen_nii = nib.load(ruta_archivo)
        # Obtener los datos de la imagen
        datos = imagen_nii.get_fdata()
        # Obtener la forma de los datos
        dimensiones = datos.shape

        # Crear una barra deslizable para la coordenida Z
        slider_z = ttk.Scale(root, from_=0, to=dimensiones[2] - 1, orient=tk.HORIZONTAL)
        slider_z.set(dimensiones[2] // 2)
        slider_z.pack(fill=tk.X)

        # Crear una barra deslizable para la coordenida Y
        slider_y = ttk.Scale(root, from_=0, to=dimensiones[1] - 1, orient=tk.HORIZONTAL)
        slider_y.set(dimensiones[1] // 2)
        slider_y.pack(fill=tk.X)

        # Crear una barra deslizable para la coordenida X
        slider_x = ttk.Scale(root, from_=0, to=dimensiones[0] - 1, orient=tk.HORIZONTAL)
        slider_x.set(dimensiones[0] // 2)
        slider_x.pack(fill=tk.X)

        # Crear una lista deslizable para seleccionar el filtro
        filtro_combobox = ttk.Combobox(root, values=["Original", "Umbralización", "Isodata", "Crecimiento de Regiones", "K-Medias"])
        filtro_combobox.set("Original")
        filtro_combobox.pack()

        def aplicar_filtro():
            filtro = filtro_combobox.get()
            if filtro == "Umbralización":
                datos_filt = umbralizacion(datos)
                guardar_imagen_umbralizada(datos_filt, "./nnn.nii")
            elif filtro == "Isodata":
                datos_filt = isodata(datos)
            elif filtro == "Crecimiento de Regiones":
                datos_filt = crecimiento_regiones(datos)
            elif filtro == "K-Medias":
                datos_filt = k_medias(datos)
            else:  # Original
                datos_filt = datos
            return datos_filt

        def guardar_imagen():
            datos_filt = aplicar_filtro()
            ruta_guardado = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")])
            if ruta_guardado:
                guardar_imagen_umbralizada(datos_filt, ruta_guardado)

        def actualizar_cortes(*args):
            datos_filt = aplicar_filtro()
            

            z = int(slider_z.get())
            y = int(slider_y.get())
            x = int(slider_x.get())

            axs[0].imshow(datos_filt[:, y, :], cmap='gray')
            axs[0].set_title(f'Corte L con y={y}, Z={z}')

            axs[1].imshow(datos_filt[x, :, :], cmap='gray')
            axs[1].set_title(f'Corte R con x={x}, Y={y}')

            axs[2].imshow(datos_filt[:, :, z], cmap='gray')
            axs[2].set_title(f'Corte L R con Z={z}')

            plt.tight_layout()
            plt.draw()


        # Asociar la función actualizar_cortes al evento de cambio en las barras deslizables
        slider_z.config(command=actualizar_cortes)
        slider_y.config(command=actualizar_cortes)
        slider_x.config(command=actualizar_cortes)
        filtro_combobox.bind("<<ComboboxSelected>>", actualizar_cortes)

        # Actualizar los cortes iniciales
        actualizar_cortes()

        boton_guardar = ttk.Button(root, text="Guardar Umbralizada", command=guardar_imagen)
        boton_guardar.pack()

# Crear un botón para cargar la imagen
boton_cargar = ttk.Button(root, text="Cargar Imagen", command=mostrar_cortes)
boton_cargar.pack()

# Agregar la figura a un lienzo de tkinter
frame = tk.Frame(root)
frame.pack()

# Agregar la figura a un lienzo de tkinter
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack()
root.mainloop()
