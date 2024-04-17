import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

def histogram_pairing(img_ref_path, img_input_path):
    # Load the NIfTI images
    img_ref = nib.load(img_ref_path)
    img_input = nib.load(img_input_path)

    # Get the image data
    img_ref_data = img_ref.get_fdata()
    img_input_data = img_input.get_fdata()

    # Compute the histograms
    hist_ref, bins_ref = np.histogram(img_ref_data, bins=50, density=True)
    hist_input, bins_input = np.histogram(img_input_data, bins=50, density=True)

    # Identify k reference points (percentiles) and their intensities in the reference histogram
    k = 10
    percentiles = np.linspace(0, 100, k+2)[1:-1]
    ref_points = np.percentile(img_ref_data, percentiles)
    ref_intensities = np.interp(ref_points, bins_ref[:-1], hist_ref)

    # Generate a piecewise function with the reference points
    ref_function = interp1d(ref_points, ref_intensities, kind='previous', fill_value='extrapolate')

    # Identify k reference points (percentiles) in the input histogram
    input_points = np.percentile(img_input_data, percentiles)

    # Map intensities according to the reference function by parts
    input_intensities = ref_function(input_points)

    # Compute the histogram of the resulting image
    img_result = np.zeros_like(img_input_data)
    for i in range(len(input_points)-1):
        img_result[(img_input_data >= input_points[i]) & (img_input_data < input_points[i+1])] = input_intensities[i]

    hist_result, bins_result = np.histogram(img_result, bins=bins_ref, density=True)

    return img_result

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI images
img_ref = nib.load('./sub-03_T1w.nii')
img_input = nib.load('./imagen.nii')

# Get the image data
img_ref_data = img_ref.get_fdata()
img_input_data = img_input.get_fdata()

# Compute the histogram pairing and the histograms of the input and resulting images
resul = histogram_pairing('./sub-03_T1w.nii', './imagen.nii')

# Display the resulting image
plt.imshow(resul[90, :, :], cmap='gray')
plt.show()