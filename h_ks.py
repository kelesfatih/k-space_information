import numpy as np
from PIL import Image
from scipy.stats import norm

# Import image as a Numpy array
image = "test.tif"
image_gs = Image.open(image).convert("L")
image_gs_array = np.array(image_gs)

# Parseval's Theorem Variance and Std
"""
!!! Caution !!!
Variance calculation depends on implementation of the FFT.
"""
variance = sum((image_gs_array.flatten() - np.mean(image_gs_array)) ** 2) / (2 * (image_gs_array.size ** 2))
std = np.sqrt(variance)

"""For calculating Hks
You need to use an actual probability and not the pdf.
Instead you need to integrate the pdf over the ends of the box. This is most easily done using the cdf.
You shouldn't have to normalize this histogram; it should sum to almost 1 on its own."""
# Probabilities
# Variance -> pdf -> divide into bins of sigma/100 -> integrate from -10sigma to +10sigma
cdf = norm.cdf(image_gs_array.flatten(), loc=0, scale=std)
bins = np.arange(cdf.min(), cdf.max() + 0.01*std, 0.01*std)
cdf_discretize = np.linspace(cdf.min(), cdf.max(), num=bins.size)
prob_entropy = cdf_discretize[0:1000]/sum(cdf_discretize[0:1000])
# Epsilon for float64: 2.220446049250313e-16 to escape log(0)
HkS = -2 * image_gs_array.size * np.sum(prob_entropy * np.log2(prob_entropy + np.finfo(float).eps))
print("HkS =", HkS)
# 147.75
