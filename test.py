import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm

# Import image as a Numpy array
image = "vertical_gradient.tif"
image_gs = Image.open(image).convert("L")
image_gs_array = np.array(image_gs)

