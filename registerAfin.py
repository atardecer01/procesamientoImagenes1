import SimpleITK as sitk
import matplotlib.pyplot as plt

def registro_imagenes(imagen_fija_path, imagen_movil_path):
    # Cargar imágenes
    imagen_fija = sitk.ReadImage(imagen_fija_path, sitk.sitkFloat32)
    imagen_movil = sitk.ReadImage(imagen_movil_path, sitk.sitkFloat32)

    # Crear registro
    registro = sitk.ImageRegistrationMethod()

    # Configurar transformación (afín)
    transformada_inicial = sitk.CenteredTransformInitializer(imagen_fija, 
                                                              imagen_movil, 
                                                              sitk.AffineTransform(3),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registro.SetInitialTransform(transformada_inicial)
    registro.SetMetricFixedMask(imagen_fija>0)

    # Configurar métrica de similitud
    registro.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registro.SetMetricSamplingStrategy(registro.RANDOM)
    registro.SetMetricSamplingPercentage(0.01)

    # Configurar optimizador
    registro.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registro.SetOptimizerScalesFromIndexShift()

    # Realizar registro
    transformada = registro.Execute(imagen_fija, imagen_movil)

    # Aplicar transformación a la imagen móvil
    imagen_movil_registrada = sitk.Resample(imagen_movil, imagen_fija, transformada, sitk.sitkLinear, 0.0, imagen_movil.GetPixelID())

    # Muestrear la imagen móvil registrada en el espacio de la imagen móvil original
    imagen_movil_registrada_resample = sitk.Resample(imagen_movil_registrada, imagen_movil)

    # Calcular la diferencia entre la imagen móvil original y la imagen móvil registrada muestreada
    diferencia = sitk.Abs(imagen_movil_registrada_resample - imagen_movil)

    return imagen_movil_registrada, diferencia


# Ejemplo de uso
imagen_fija_path = "./imagen.nii"
imagen_movil_path = "./imagen_rotada.nii"
imagen_movil_registrada, diferencia = registro_imagenes(imagen_fija_path, imagen_movil_path)

# Guardar la imagen registrada
sitk.WriteImage(imagen_movil_registrada, "./imagen_movil_registrada.nii")

# Visualizar la diferencia entre la imagen móvil original y la imagen móvil registrada
plt.imshow(sitk.GetArrayFromImage(diferencia)[:,:,0], cmap='gray')
plt.title('Diferencia entre la imagen móvil original y la registrada')
plt.axis('off')
plt.show()