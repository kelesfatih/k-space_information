from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# convert and save images in grayscale
img_obj_gs = Image.open("C1-confocal-series-0025.tif").convert("L")
img_obj_gs_arr = np.array(img_obj_gs)

img_obj_gs_arr_fft = np.fft.fft2(img_obj_gs_arr)


frequency_domain = np.fft.fft2(img_obj_gs_arr)
frequency_domain_shifted = np.fft.fftshift(frequency_domain)
energy_spatial = np.sum(np.square(img_obj_gs_arr))
energy_frequency = np.sum(np.square(np.abs(frequency_domain_shifted)))
pdf_spatial = np.square(img_obj_gs_arr) / energy_spatial

plt.hist(pdf_spatial.ravel(), bins=100, density=True, color='blue', alpha=0.7)
plt.title('Probability Density Function (PDF) in the Spatial Domain')
plt.xlabel('Pixel Value')
plt.ylabel('Probability Density')
plt.show()

if np.isclose(energy_spatial, energy_frequency):
    print("Parseval's Theorem holds: Energy in spatial and frequency domains is equal.")
else:
    print("Parseval's Theorem does not hold: Energy in spatial and frequency domains is not equal.")
