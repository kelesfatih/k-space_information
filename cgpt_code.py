import numpy as np
from scipy.fft import fftn, fftshift
from scipy.stats import entropy
from PIL import Image
import pywt

def convert_to_grayscale(image_path):
    img = Image.open(image_path)
    gray_img = img.convert('L')
    return np.array(gray_img)

def calculate_entropy(gray_img):
    wt = pywt.dwt2(gray_img, 'haar')
    # Handling potential division by zero and logarithm of zero
    entropy_value = np.sum(np.nan_to_num(wt[0]**2 * np.log(np.nan_to_num(wt[0]**2))))
    return entropy_value

def calculate_epd_entropy(gray_img, block_size=16):
    img_height, img_width = gray_img.shape
    num_blocks_x = img_width // block_size
    num_blocks_y = img_height // block_size
    entropies = []
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = gray_img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            # Handling potential division by zero and logarithm of zero
            entropy_value = np.sum(np.nan_to_num(block**2 * np.log(np.nan_to_num(block**2))))
            entropies.append(entropy_value)
    average_entropy = np.nanmean(entropies)  # Using np.nanmean to handle empty slices
    return average_entropy

def calculate_ksi(gray_img):
    fft_img = fftshift(fftn(gray_img))
    magnitude_fft = np.abs(fft_img)
    normalized_magnitude_fft = magnitude_fft / np.sum(magnitude_fft)
    # Handling potential division by zero and logarithm of zero
    ksi_value = -np.sum(np.nan_to_num(normalized_magnitude_fft * np.log(np.nan_to_num(normalized_magnitude_fft))))
    return ksi_value

def calculate_ksi_with_epd(gray_img):
    fft_img = fftshift(fftn(gray_img))
    magnitude_fft = np.abs(fft_img)
    normalized_magnitude_fft = magnitude_fft / np.sum(magnitude_fft)
    # Calculate EPD: Probability distribution of magnitude values
    epd = np.histogram(normalized_magnitude_fft.flatten(), bins=100, density=True)[0]
    # Handling potential division by zero and logarithm of zero
    ksi_value = -np.sum(np.nan_to_num(epd * np.log(np.nan_to_num(epd))))
    return {"ksi": ksi_value, "epd": epd}

# Example usage
image_path = "C:/Users/fatih/OneDrive/Desktop/entropyimage/testentro_32_t.tif"

# Convert to grayscale using Pillow
gray_img = convert_to_grayscale(image_path)

# Calculate entropy
entropy_value = calculate_entropy(gray_img)
print("Entropy value:", entropy_value)

# Calculate EPD-based entropy
epd_entropy_value = calculate_epd_entropy(gray_img)
print("EPD-based Entropy value:", epd_entropy_value)

# Calculate k-space information (kSI)
ksi_value = calculate_ksi(gray_img)
print("kSI value:", ksi_value)

# Calculate k-space information (kSI) with Experimental Probability Distribution (EPD)
result = calculate_ksi_with_epd(gray_img)
print("kSI value:", result["ksi"])
print("Experimental Probability Distribution (EPD):")
print(result["epd"])
