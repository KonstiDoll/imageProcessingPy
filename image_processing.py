import base64
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import random
from PIL import Image, ImageDraw
from svg.path import Path, Line
import svgwrite
import math

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

def find_contours_and_generate_gcode1(image):
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
        gcode = generate_gcode(contours)
    return gcode, contour_image

def find_contours_and_generate_gcode(image):
    # Load image and convert to grayscale
    # if len(image.shape) == 3:
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # else:
    #     gray = image

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize G-code string
    gcode = "G90\nG21\nG1 Z0.0 F300\n"

    # Iterate over contours
    for contour in contours:
        for i, point in enumerate(contour):
            x, y = point[0]
            if i == 0:
                gcode += f"G0 X{x} Y{y}\nG1 Z-0.1 F300\n"
            else:
                gcode += f"G1 X{x} Y{y} F100\n"

        gcode += "G1 Z0.0 F300\n"
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        contour_image = cv2.drawContours(img, [contour], -1, color, 1)
    # Draw contours on the original image
    #contour_image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    return gcode, Image.fromarray(contour_image)

def generate_gcode(contours, feed_rate=3000, pen_up_height=2, pen_down_height=0, header='', footer=''):
    gcode = []

    # Add the header
    gcode.append(header)

    for contour in contours:
        # Move to the start point of the contour
        start_point = contour[0][0]
        gcode.append(f'G0 X{start_point[0]} Y{start_point[1]} Z{pen_up_height}')
        gcode.append(f'G1 Z{pen_down_height} F{feed_rate}')  # Lower the pen (pen down)

        # Follow the contour points
        for point in contour[1:]:
            point = point[0]
            gcode.append(f'G1 X{point[0]} Y{point[1]} F{feed_rate}')

        gcode.append(f'G1 Z{pen_up_height} F{feed_rate}')  # Raise the pen (pen up)

    # Add the footer
    gcode.append(footer)

    # Join the G-code lines into a single string
    gcode_str = '\n'.join(gcode)
    return gcode_str

def apply_horiz_line_filling(input_image, line_spacing):
    # Convert the NumPy ndarray to a Pillow Image object
    img = Image.fromarray(input_image.astype('uint8')).convert('1')
    
    # Create an empty path
    path = Path()
    
    # Draw horizontal lines with specified line_spacing
    for y in range(0, img.size[1], line_spacing):
        start_x = None
        for x in range(img.size[0]):
            if img.getpixel((x, y)) == 0:  # If the input image pixel is black
                if start_x is None:
                    start_x = x
            elif start_x is not None:
                path.append(Line(start=(start_x, y), end=(x-1, y)))
                start_x = None

    # Create an SVG drawing and add the path to it
    output_path = './output.svg'
    svg_drawing = svgwrite.Drawing(output_path, size=img.size)
    path_element = svg_drawing.add(svg_drawing.path(fill="none", stroke="black", stroke_width="1"))
    for segment in path:
        if segment.start != segment.end:
            path_element.push("M", segment.start, "L", segment.end)

    # Save the SVG file
    svg_drawing.save()

    with open(output_path, 'rb') as f:
        svg_data = f.read()
    svg_data_uri = 'data:image/svg+xml;base64,' + base64.b64encode(svg_data).decode('utf-8')

    import math

import math

def generate_spiral_path(image):
    # Detect contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Generate the spiral path for the contour
    points = generate_spiral_points(contour)
    spiral_svg = generate_spiral_svg(points)

    # Save the SVG file
    output_path = './output.svg'
    with open(output_path, 'w') as f:
        f.write(spiral_svg)

    # Return the SVG path string
    return [spiral_svg]

def generate_spiral_points(center_x, center_y, radius, num_points):
    """Generate `num_points` equidistant points along a spiral path starting from `(center_x, center_y)`."""
    points = []
    angle_step = 2 * math.pi / num_points  # The angle increment between points
    current_radius = 0  # The current radius of the spiral
    for i in range(num_points):
        angle = i * angle_step
        current_radius += radius * 0.02  # Increase the radius by a fixed factor each step
        x = center_x + current_radius * math.cos(angle)
        y = center_y + current_radius * math.sin(angle)
        points.append((x, y))
    return points

import svgwrite

def generate_spiral_svg(points):
    path = svgwrite.path.Path(fill='none', stroke='black', stroke_width=1)

    # Get the starting point of the spiral
    start = points[0]

    # Define the radius and step size of the spiral
    radius = 5
    step = 1

    # Move to the starting point
    path.push('M', start)

    # Follow the contour with a spiral path
    for i in range(len(points)):
        # Get the current point
        point = points[i]

        # Calculate the distance between the current point and the starting point
        distance = ((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5

        # Calculate the angle of the spiral
        angle = (distance / radius) * 2 * 3.14

        # Calculate the new position of the spiral
        x = start[0] + radius * (angle * 0.5 / 3.14) * (point[1] - start[1]) / distance
        y = start[1] + radius * (angle * 0.5 / 3.14) * (point[0] - start[0]) / distance

        # Draw a line to the new position
        path.push('L', (x, y))

        # Update the starting point and radius for the next step
        start = (x, y)
        radius += step

    # Return the SVG path string
    return path.tostring()


    # Return the blended image
    return [svg_data_uri]