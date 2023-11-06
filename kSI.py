import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm

# Import image as a Numpy array
image = "vertical_gradient.tif"
image_gs = Image.open(image).convert("L")
image_gs_array = np.array(image_gs)

# Parseval's Theorem Variance and Std
variance = sum((image_gs_array.flatten() - np.mean(image_gs_array)) ** 2) / (2 * (image_gs_array.size ** 2))
std = np.sqrt(variance)

# Bins for PDF
range_min = -10 * std
range_max = 10 * std
bin_width = std / 100
x = np.arange(range_min, range_max + bin_width, bin_width)
# PDF
pdf = norm.pdf(x, scale=std)
# Normalise PDF
pdf = pdf / np.sum(pdf)
# HkS
HkS = -2 * image_gs_array.size * np.sum(pdf * np.log2(pdf))
print("HkS =", HkS)

# FFT
image_fft = np.fft.fft2(image_gs_array)
image_fft_real = np.real(image_fft)
image_fft_imaginary = np.imag(image_fft)

# IkS
cdf = np.cumsum(pdf)  # cumulative sum for integration
image_fft_real_probabilities = []
for c in image_fft_real.flatten()[1:]:  # del DC comp by [1:]
    lower_bound = c - (0.005 * std)
    upper_bound = c + (0.005 * std)
    lower_idx = np.searchsorted(x, lower_bound)  # how to index or what bin I should index
    upper_idx = np.searchsorted(x, upper_bound)
    integrated_probability = cdf[upper_idx] - cdf[lower_idx]
    image_fft_real_probabilities.append(integrated_probability)

image_fft_imaginary_probabilities = []
for c in image_fft_imaginary.flatten()[1:]:
    lower_bound = c - (0.005 * std)
    upper_bound = c + (0.005 * std)
    lower_idx = np.searchsorted(x, lower_bound)
    upper_idx = np.searchsorted(x, upper_bound)
    integrated_probability = cdf[upper_idx] - cdf[lower_idx]
    image_fft_imaginary_probabilities.append(integrated_probability)

IkS = sum(-np.log2(image_fft_real_probabilities) - np.log2(image_fft_imaginary_probabilities))
print("IkS =", IkS)

# kSI
kSI = HkS - IkS
print("kSI =", kSI)
