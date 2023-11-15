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

# 2D Inverse FFT
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
# CDF from c - 0.005*sigma to c + 0.005*sigma for c = a or b
# The distribution should be scaled to standard deviation
prob_real = (norm.cdf(image_fft_real.flatten() + (0.005 * std), loc=0, scale=std) -
             norm.cdf(image_fft_real.flatten() - (0.005 * std), loc=0, scale=std))
prob_imag = (norm.cdf(image_fft_imaginary.flatten() + (0.005 * std), loc=0, scale=std) -
             norm.cdf(image_fft_imaginary.flatten() - (0.005 * std), loc=0, scale=std))

# [1: ] was used to remove a DC component of real and imaginary part
# Epsilon was used to escape log(0) where float64: 2.220446049250313e-16
IkS = np.sum(- np.log2(prob_real[1:] + np.finfo(np.float64).eps) - np.log2(prob_imag[1:] + np.finfo(np.float64).eps))
print("IkS =", IkS)

"""
HkS
"""
# Probabilities
cdf_image = (norm.cdf(image_gs_array.flatten() + (std * 0.005), loc=0, scale=std) -
             norm.cdf(image_gs_array.flatten() - (std * 0.005), loc=0, scale=std))
bin_array = np.arange(image_gs_array.min(), image_gs_array.max() + 0.01 * std, 0.01 * std)
discretized_cdf = np.linspace(cdf_image.min(), cdf_image.max(), bin_array.size)[np.searchsorted(bin_array, - 10 * std):
                                                                                np.searchsorted(bin_array, 10 * std)]
prob_entropy = discretized_cdf/sum(discretized_cdf)
HkS = -2 * image_gs_array.size * np.sum(prob_entropy * np.log2(prob_entropy + np.finfo(float).eps))
print("HkS =", HkS)
# 147.75

"""
kSI
"""
kSI = HkS - IkS
print("kSI =", kSI)
