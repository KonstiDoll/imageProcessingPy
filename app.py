from flask import Flask, request, jsonify
import cv2
import base64
import io
from PIL import Image
import numpy as np
from image_processing import (
    apply_grayscale, apply_canny, apply_color_quantization, apply_custom_color_quantization, apply_find_contours, find_contours_and_generate_gcode,
    apply_horiz_line_filling)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    if request.method == "POST":
        
        
        # Read the image from the request
        image_data = request.files["image"].read()
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Read the processing method from the request
        processing_method = request.form["processing_method"]

        if request.form.get("submit_type") == "generate_gcode":
            # Generate G-code and contour image
            gcode, contour_image = find_contours_and_generate_gcode(image)

            # Save the contour image as base64 string
            buffered = io.BytesIO()
            contour_image.save(buffered, format="PNG")
            contour_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Return the contour image and G-code
            return jsonify({'contour_image_base64': contour_image_base64, 'gcode': gcode})
        processed_images = []
        processed_svgs = []
        # Apply the chosen processing method
        if processing_method == "grayscale":
            processed_images = [apply_grayscale(image)]
        elif processing_method == "canny":
            processed_images = [apply_canny(image)]
        elif processing_method == "color_quantization":
            num_colors = int(request.form.get("num_colors", 3))
            quantized_image, binary_images = apply_color_quantization(image, num_colors)
            processed_images = [quantized_image] + binary_images

        elif processing_method == "custom_color_quantization":
            num_colors = int(request.form["num_colors"])
            colors = [tuple(int(request.form[f"color_{i}"][j:j+2], 16) for j in (1, 3, 5)) for i in range(num_colors)]
            processed_images = apply_custom_color_quantization(image, colors)
        elif processing_method == "find_Contours":
            processed_images = apply_find_contours(image)
        elif processing_method == "horizontal_line_filling":
            line_spacing = int(request.form["line_spacing"])
            processed_svgs = apply_horiz_line_filling(image, line_spacing)
        else:
            processed_images = [image]


        # Convert the processed images back to PNG format and generate the HTML output
        output = ""
        for processed_svg in processed_svgs:
            output+= f'<img src="{ processed_svg }" alt="SVG Image">'

        for processed_image in processed_images:
            # if isinstance(processed_image, np.ndarray) and len(processed_image.shape) == 2:  # Grayscale image
            #     output_image = Image.fromarray(processed_image, mode='L')
            # else:  # Color image
            output_image = Image.fromarray(processed_image)
            output_buffer = io.BytesIO()
            output_image.save(output_buffer, format="PNG")
            output_data = base64.b64encode(output_buffer.getvalue()).decode("ascii")
            output += f'<img src="data:image/png;base64,{output_data}" /><br>'

    
    return f'''
<form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <select name="processing_method">
        <option value="original">Original</option>
        <option value="grayscale">Grayscale</option>
        <option value="canny">Canny Edge Detection</option>
        <option value="color_quantization">Color Quantization</option>
        <option value="custom_color_quantization">Custom Color Quantization</option>
        <option value="find_Contours">Find Contours</option>
        <option value="horizontal_line_filling">Fill Horizontally</option>
    </select>
    <input type="number" name="num_colors" min="1" max="10" value="3" id="numColorsQuant" style="display:none;" required>
    <input type="number" name="line_spacing" min="1" max="100" value="3" id="line_spacing" style="display:none;" required>
    <input type="number" name="num_colors" min="1" max="10" value="3" id="numColors" style="display:none;" required>
    <div id="colorPickers" style="display:none;"></div>
    <input type="hidden" name="submit_type" value="process_image">
    <button type="submit">Submit</button>
    <input type="hidden" name="submit_type" value="generate_gcode">
    <button type="submit" name="generate_gcode">Generate G-code</button>

</form>

<div id="contour-images"></div>


<script src="/static/main.js"></script>
''' + output
# <canvas id="gcodeCanvas" width="1000" height="1000" style="border:1px solid #000000;"></canvas>

# <textarea id="gcode-display" readonly></textarea>
@app.route('/generate_gcode', methods=['POST'])
def generate_gcode():
    # Read the image from the request
    image_data = request.files["image"].read()
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

    gcode, contour_image = find_contours_and_generate_gcode(image)
    contour_image_pil = contour_image
    buffer = io.BytesIO()
    contour_image_pil.save(buffer, format="PNG")
    contour_image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return jsonify({
        'gcode': gcode,
        'contour_image': contour_image_base64
    })


if __name__ == "__main__":
    app.run(debug=True)
