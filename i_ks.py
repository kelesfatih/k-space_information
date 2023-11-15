import numpy as np
from PIL import Image
from scipy.stats import norm

# Import image as a Numpy array
image = "test.tif"
image_gs = Image.open(image).convert("L")
image_gs_array = np.array(image_gs)

# FFT
image_fft = np.fft.ifft2(image_gs_array)
image_fft_real = np.real(image_fft)
image_fft_imaginary = np.imag(image_fft)

# Parseval's Theorem Variance and Std
"""
!!! Caution !!!
Variance calculation depends on implementation of the FFT.
"""
variance = sum((image_gs_array.flatten() - np.mean(image_gs_array)) ** 2) / (2 * (image_gs_array.size ** 2))
std = np.sqrt(variance)

# Probabilities

# CDF from c - 0.005*sigma to c + 0.005*sigma for c = a or b
# The distribution should be scaled to standard deviation.
prob_real = (norm.cdf(image_fft_real.flatten() + (0.005 * std), loc=0, scale=std)
             - norm.cdf(image_fft_real.flatten() - (0.005 * std), loc=0, scale=std))
prob_imag = (norm.cdf(image_fft_imaginary.flatten() + (0.005 * std), loc=0, scale=std)
             - norm.cdf(image_fft_imaginary.flatten() - (0.005 * std), loc=0, scale=std))

# [1: ] is to remove a DC component of real and imaginary part.
# Epsilon for float64: 2.220446049250313e-16 to escape log(0)
IkS = np.sum(- np.log2(prob_real[1:] + np.finfo(np.float64).eps) - np.log2(prob_imag[1:] + np.finfo(np.float64).eps))
print("IkS =", IkS)
