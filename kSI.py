from PIL import Image
import numpy as np

image = "test.tif"
img_obj_gs = Image.open(image).convert("L")
img_obj_gs_arr = np.array(img_obj_gs)

# Parseval's Theorem Variance
variance = np.sqrt((sum(np.square(img_obj_gs_arr.flatten() - np.mean(img_obj_gs_arr)))) /
                   (2 * np.square(len(img_obj_gs_arr.flatten()))))
# HkS

# IkS

# kSI
