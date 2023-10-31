from PIL import Image
import numpy as np

# Import image to environment
image = "test.tif"
img_obj_gs = Image.open(image).convert("L")
img_obj_gs_arr = np.array(img_obj_gs)

# Parseval's Theorem Variance (sigma^2) and Standard Deviation (sigma)
variance = ((sum(np.square(img_obj_gs_arr.flatten() - np.mean(img_obj_gs_arr)))) /
            (2 * np.square(len(img_obj_gs_arr.flatten()))))
std = np.sqrt(variance)

# HkS
# -10sigma to 10sigma with 0.01sigma intervals
range_min = -10 * std
range_max = 10 * std
bin_width = std / 100
x = np.arange(range_min, range_max + bin_width, bin_width)
# pdf
pdf = (1 / (2 * np.pi * variance)) * np.exp(-(np.square(x) / (2 * variance)))
pdf = pdf / np.sum(pdf)
p_c = 1  # couldn't find to calculate a way
H_kS = -2 * np.sum(p_c * np.log2(p_c))  # 147.75
print("HkS =", H_kS)

# IkS
f_image = np.fft.ifft2(img_obj_gs_arr)
f_image_real = np.real(f_image)
f_image_imaginary = np.imag(f_image)
real_probabilities = pdf[np.searchsorted(x, f_image_real)]
imaginary_probabilities = pdf[np.searchsorted(x, f_image_imaginary)]
dc_deleted_real_p = real_probabilities.flatten()[1:]
dc_deleted_imaginary_p = imaginary_probabilities.flatten()[1:]
I_kS = sum(-np.log2(dc_deleted_real_p) - np.log2(dc_deleted_imaginary_p))
print("IkS =", I_kS)

# kSI = HkS - IkS
k_SI = H_kS - I_kS
print("kSI =", k_SI)

