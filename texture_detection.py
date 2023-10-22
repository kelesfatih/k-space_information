import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

img = np.array(Image.open("C1-confocal-series-0025.tif").convert("L"))
image = img_as_ubyte(img)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                               sharex=True, sharey=True)

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("Image")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
ax1.set_title("Entropy")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

fig.tight_layout()

plt.show()