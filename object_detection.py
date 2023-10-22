import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skimage.filters.rank import entropy
from skimage.morphology import disk

img_obj = Image.open("C1-confocal-series-0025.tif")
img_obj_gs = Image.open("C1-confocal-series-0025.tif").convert("L")

img = np.array(img_obj)
entr_img = entropy(img, disk(10))

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

img0 = ax0.imshow(img_obj)
ax0.set_title("Object")
ax1.imshow(img_obj_gs, cmap='gray')
ax1.set_title("Greyscale")
ax2.imshow(entr_img, cmap='viridis')
ax2.set_title("Local entropy")

fig.tight_layout()
fig.colorbar(img0, ax=ax2)

plt.show()
