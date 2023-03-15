import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import random

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)
    return cv2.Canny(gray_image, 0, 1000)

def apply_color_quantization(image, n_clusters=3):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters)
    kmeans.fit(pixels)

    # Replace each pixel with its corresponding cluster centroid
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_pixels = np.round(quantized_pixels).astype(np.uint8)

    # Reshape the quantized pixels back to the original image dimensions
    quantized_image = quantized_pixels.reshape(image.shape)

    # Create binary images for each color in the quantized image
    unique_colors = np.unique(quantized_pixels, axis=0)
    binary_images = []
    for color in unique_colors:
        binary_image = 255 - (np.all(quantized_image == color, axis=-1).astype(np.uint8) * 255)
        binary_images.append(binary_image)

    return quantized_image, binary_images

def apply_custom_color_quantization(image, colors):
    colors = np.array(colors, dtype=np.uint8)
    tree = KDTree(colors)

    # Flatten the image array to a 2D array of pixels
    height, width, channels = image.shape
    flattened_image = image.reshape(-1, channels)

    # Find the index of the closest color for each pixel
    _, closest_color_idx = tree.query(flattened_image)

    # Replace each pixel with the closest color
    quantized_pixels = colors[closest_color_idx]
    
    # Reshape the quantized pixels back to the original image dimensions
    quantized_image = quantized_pixels.reshape(height, width, channels)

    return [quantized_image]

def apply_find_contours(image):
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image


    # Find contours
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a blank image
    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        print(color)
        cv2.drawContours(contour_image, [contour], -1, color, 1)
    
    return [contour_image]

