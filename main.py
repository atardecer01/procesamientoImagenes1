import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
# Funciones de procesamiento de imágenes
from algoritmos.isodata import isodata
from algoritmos.umbralizacion import umbralizacion
from algoritmos.crecimientoR import crecimiento_regiones
from algoritmos.k_medias import k_medias

# Variable global para almacenar la imagen cargada
datos = None
dimensiones = [100, 100, 100]
seleccionando = False
x_ini, y_ini, x_fin, y_fin = 0, 0, 0, 0

def cargar_imagen():
    global datos, dimensiones
    ruta_archivo = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")])
    if ruta_archivo:
        # Cargar la imagen .nii
        imagen_nii = nib.load(ruta_archivo)
        datos = imagen_nii.get_fdata()
        dimensiones = datos.shape
        print(f"Imagen cargada: {ruta_archivo}")

        # Actualizar los sliders con las nuevas dimensiones
        slider_x.config(to=dimensiones[0] - 1)
        slider_y.config(to=dimensiones[1] - 1)
        slider_z.config(to=dimensiones[2] - 1)

        # Asociar la función actualizar_cortes al evento de cambio en las barras deslizables
        slider_z.config(command=actualizar_cortes)
        slider_y.config(command=actualizar_cortes)
        slider_x.config(command=actualizar_cortes)

        # Habilitar el botón "Aplicar filtro"
        button.config(state=tk.NORMAL)

        # Mostrar la imagen en los cortes iniciales
        actualizar_cortes()

def actualizar_cortes(*arg):
    z = int(slider_z.get())
    y = int(slider_y.get())
    x = int(slider_x.get())
    slider_z_label.config(text="Coordenada de Z= " + str(z))
    slider_y_label.config(text="Coordenada de Y= " + str(y))
    slider_x_label.config(text="Coordenada de X= " + str(x))

    # Obtener el filtro seleccionado
    filtro = filtro_combobox.get()
    if filtro == "Umbralización":
        datos_filt = umbralizacion(datos)
    elif filtro == "Isodata":
        datos_filt = isodata(datos)
    elif filtro == "Crecimiento de Regiones":
        datos_filt = crecimiento_regiones(datos)
    elif filtro == "K-Medias":
        datos_filt = k_medias(datos)
    else:  # Original
        datos_filt = datos

    # Actualizar las imágenes en las subparcelas
    axs[0, 0].imshow(datos[:, y, :], cmap='gray')
    axs[0, 0].set_title(f'Corte L con y={y}')

    axs[0, 1].imshow(datos[x, :, :], cmap='gray')
    axs[0, 1].set_title(f'Corte R con x={x}')

    axs[0, 2].imshow(datos[:, :, z], cmap='gray')
    axs[0, 2].set_title(f'Corte L R con Z={z}')

    axs[1, 0].imshow(datos_filt[:, y, :], cmap='gray')
    axs[1, 0].set_title(f'Filtro: {filtro}')

    axs[1, 1].imshow(datos_filt[x, :, :], cmap='gray')
    axs[1, 1].set_title(f'Filtro: {filtro}')

    axs[1, 2].imshow(datos_filt[:, :, z], cmap='gray')
    axs[1, 2].set_title(f'Filtro: {filtro}')

    plt.tight_layout()
    canvas.draw()

def activar_seleccion(event):
    global seleccionando, x_ini, y_ini
    seleccionando = True
    x_ini, y_ini = event.x, event.y

def seleccionar_area(event):
    global x_fin, y_fin
    x_fin, y_fin = event.x, event.y
    canvas.delete("selection_rectangle")
    canvas.create_rectangle(x_ini, y_ini, x_fin, y_fin, outline="red", tag="selection_rectangle")

def guardar_area():
    global seleccionando, x_ini, y_ini, x_fin, y_fin
    seleccionando = False
    area_seleccionada = datos[y_ini:y_fin, x_ini:x_fin]
    imagen_recortada = Image.fromarray(area_seleccionada)
    ruta_guardar = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if ruta_guardar:
        imagen_recortada.save(ruta_guardar)
        print(f"Área seleccionada guardada en: {ruta_guardar}")


# Crear la ventana
root = tk.Tk()
root.title("Interfaz con Barra Lateral")

# Configurar el tamaño de la ventana
root.geometry("1200x800")

# Crear el frame principal
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Crear la barra lateral
sidebar_frame = ttk.Frame(main_frame, width=100, relief=tk.RAISED, padding=(10, 10))
sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)


def aplicar_filtro():
    # Obtener el filtro seleccionado
    filtro = filtro_combobox.get()
    if filtro == "Umbralización":
        datos_filt = umbralizacion(datos)
    elif filtro == "Isodata":
        datos_filt = isodata(datos)
    elif filtro == "Crecimiento de Regiones":
        datos_filt = crecimiento_regiones(datos)
    elif filtro == "K-Medias":
        datos_filt = k_medias(datos)
    else:  # Original
        datos_filt = datos

    # Actualizar las imágenes en las subparcelas con el nuevo filtro
    actualizar_cortes()




filtro_label = ttk.Label(sidebar_frame, text="Escoge un filtro de segmentacion")
# Crear una lista deslizable para seleccionar el filtro
filtro_combobox = ttk.Combobox(sidebar_frame, values=["Original", "Umbralización", "Isodata", "Crecimiento de Regiones", "K-Medias"])
filtro_combobox.set("Original")
filtro_label.pack(pady=10)
filtro_combobox.pack()

# Crear un botón en la barra lateral
button = ttk.Button(sidebar_frame, text="Aplicar filtro", command=aplicar_filtro, state=tk.DISABLED)
button.pack(pady=50)


# Crear una barra deslizable para la coordenida Z
slider_z = ttk.Scale(sidebar_frame, from_=0, to=dimensiones[2] - 1, orient=tk.HORIZONTAL)
slider_z_label = ttk.Label(sidebar_frame, text="Coordenada de Z= "+str(slider_z.get()))
slider_z.set(dimensiones[2] // 2)
slider_z_label.pack()
slider_z.pack()


# Crear una barra deslizable para la coordenida Y
slider_y = ttk.Scale(sidebar_frame, from_=0, to=dimensiones[1] - 1, orient=tk.HORIZONTAL)
slider_y.set(dimensiones[1] // 2)
slider_y_label = ttk.Label(sidebar_frame, text="Coordenada de Y= "+str(slider_y.get()))
slider_y_label.pack()
slider_y.pack()


# Crear una barra deslizable para la coordenida X
slider_x = ttk.Scale(sidebar_frame, from_=0, to=dimensiones[0] - 1, orient=tk.HORIZONTAL)
slider_x.set(dimensiones[0] // 2)
slider_x_label = ttk.Label(sidebar_frame, text="Coordenada de X= "+str(slider_x.get()))
slider_x_label.pack()
slider_x.pack()




# Crear el contenido principal
content_frame = ttk.Frame(main_frame, padding=(10, 10))
content_frame.pack(fill=tk.BOTH, expand=True)

# Crear un botón en el contenido principal para cargar una imagen .nii
button_cargar_imagen = ttk.Button(content_frame, text="Cargar Imagen .nii", command=cargar_imagen)
button_cargar_imagen.pack()

# Crear una figura de matplotlib
# Crear una nueva figura con 6 subparcelas
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
fig.tight_layout(pad=3.0)

# Mostrar la figura en un lienzo de tkinter
canvas = FigureCanvasTkAgg(fig, master=content_frame)
canvas.get_tk_widget().pack()

# Enlazar el evento de clic en la imagen a la función activar_seleccion
canvas.mpl_connect('button_press_event', activar_seleccion)

# Mostrar la ventana
root.mainloop()
