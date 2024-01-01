import numpy as np
from PIL import Image
from scipy.stats import norm

# Image name
image = "test.tif"
# L = R * 299/1000 + G * 587/1000 + B * 114/1000
image_gs = Image.open(image).convert("L")
# Import image as a Numpy array
image_gs_array = np.array(image_gs)
# Save the grayscale image
# If your file extension is tiff instead of tif change -4 to -5
image_gs.save(f"{image[:-4]}_gs.tif", format="TIFF")

# 2D FFT
image_fft = np.fft.ifft2(image_gs_array)
# Real and imaginary parts of FFT
image_fft_real = np.real(image_fft)
image_fft_imaginary = np.imag(image_fft)

# Parseval's Theorem Variance and Std !!! Caution !!! Variance calculation depends on the implementation of the FFT
variance = np.sum((image_gs_array.flatten() - np.mean(image_gs_array)) ** 2) / (2 * (image_gs_array.size ** 2))
std = np.sqrt(variance)

"""
IkS
"""
# Probabilities
prob_real = (norm.cdf(image_fft_real.flatten() + (0.005 * std), loc=0, scale=std) -
             norm.cdf(image_fft_real.flatten() - (0.005 * std), loc=0, scale=std))[1:]
prob_imag = (norm.cdf(image_fft_imaginary.flatten() + (0.005 * std), loc=0, scale=std) -
             norm.cdf(image_fft_imaginary.flatten() - (0.005 * std), loc=0, scale=std))[1:]

log_real = - np.log2(prob_real)
log_imag = - np.log2(prob_imag)

log_real[log_real == np.inf] = np.nan
log_imag[log_imag == np.inf] = np.nan
IkS = np.nansum(log_real + log_imag)
print("IkS =", IkS)

"""
HkS
"""
# Probabilities
hist, _ = np.histogram(image_gs_array.flatten(), bins=2, range=(0, 2))
normalized_hist = hist / image_gs_array.size
HkS = -2 * image_gs_array.size * np.nansum(normalized_hist * np.log2(normalized_hist))
print("HkS =", HkS)

"""
kSI
"""
kSI = HkS - IkS
print("kSI =", kSI)
