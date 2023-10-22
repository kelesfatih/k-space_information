from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# import sys
# np.set_printoptions(threshold=sys.maxsize)


def entropy(signal):
    # function returns entropy of a signal, and signal must be a 1-D numpy array
    lensig = signal.size
    symset = list(set(signal))
    numsym = len(symset)
    propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent


colorIm = Image.open("confocal-series.tif")
greyIm = colorIm.convert('L')

colorIm_array = np.array(colorIm)
greyIm_array = np.array(greyIm)

# For size_of_region = 5 the region contains 10*10 = 100 pixel values.
size_of_region = 5
shape_of_greyIM = greyIm_array.shape
entropy_array = greyIm_array.copy()

for row in range(shape_of_greyIM[0]):
    for col in range(shape_of_greyIM[1]):
        L_x = np.max([0, col - size_of_region])
        U_x = np.min([shape_of_greyIM[1], col + size_of_region])
        L_y = np.max([0, row - size_of_region])
        U_y = np.min([shape_of_greyIM[0], row + size_of_region])
        region = greyIm_array[L_y:U_y, L_x:U_x].flatten()
        entropy_array[row, col] = entropy(region)


plt.subplot(1,3,1)
plt.imshow(colorIm)

plt.subplot(1,3,2)
plt.imshow(greyIm, cmap=plt.cm.gray)

plt.subplot(1,3,3)
plt.imshow(entropy_array, cmap=plt.cm.jet)

plt.xlabel('Entropy in 10x10 neighbourhood')
plt.colorbar()

plt.show()
