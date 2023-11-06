import numpy as np
from PIL import Image
from scipy.stats import norm
from scipy.integrate import quad

# Import image as a Numpy array
# 1. I imported the image as a grayscale and converted to Numpy array
image = "test.tif"
image_gs = Image.open(image).convert("L")
image_gs_array = np.array(image_gs)

# FFT
# 2. I calculated 2D FFT of the array and seperate real and imaginary parts.
image_fft = np.fft.fft2(image_gs_array)
image_fft_real = np.real(image_fft)
image_fft_imaginary = np.imag(image_fft)

# Parseval's Theorem Variance and Std
# 3. I calculated variance and std with Parseval's equation.
variance = sum((image_gs_array.flatten() - np.mean(image_gs_array)) ** 2) / (2 * (image_gs_array.size ** 2))
std = np.sqrt(variance)


# Probabilities
# 4. I defined an function named "pdf_integrator" the equation is from 4th page of suplementary file under
# "Effect of integration window on information computation" title.
def pdf_integrator(x):
    return (1 / (2 * np.pi * variance)) * (np.exp(-(x ** 2) / (2 * variance)))


# 5. I passed every value in real and imaginary parts to quad function to integrate
# from c - 0.005*sigma to c + 0.005*sigma and stored in seperate lists.
image_fft_real_probabilities = []
for i in image_fft_real.flatten():
    p = quad(pdf_integrator, i - (std * 0.005), i + (std * 0.005))
    image_fft_real_probabilities.append(p[0])

image_fft_imaginary_probabilities = []
for i in image_fft_imaginary.flatten():
    p = quad(pdf_integrator, i - (std * 0.005), i + (std * 0.005))
    image_fft_imaginary_probabilities.append(p[0])

# 6. I summed -log2 of probabilities with excluding first element which is DC component.
IkS = sum(- np.log2(image_fft_real_probabilities[1:]) - np.log2(image_fft_imaginary_probabilities[1:]))
print("IkS =", IkS)


# Bins for PDF
# 7. I defined the range for HkS computation from -10*sigma to +10*sigma with 0.01sigma steps.
# I create PDF with scaling standard deviation and normalize the PDF function.
range_min = -10 * std
range_max = 10 * std
bin_width = std / 100
bin_array = np.arange(range_min, range_max + bin_width, bin_width)
# PDF
pdf = norm.pdf(bin_array, loc=0, scale=std)
# Normalise PDF
pdf = pdf / np.sum(pdf)
# HkS
# 8. I tried to calculate entropy as you described in paper " -2 * N of pixels * sum(pdf * log2(pdf)"
HkS = -2 * image_gs_array.size * np.sum(pdf * np.log2(pdf))
print("HkS =", HkS)

# kSI
kSI = HkS - IkS
print("kSI =", kSI)
