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
H_kS = -2 * np.sum(pdf * np.log2(pdf))
print(H_kS)

# IkS

# kSI = HkS - IkS
