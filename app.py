from flask import Flask, request
import cv2
import base64
import io
from PIL import Image
import numpy as np
from image_processing import (
    apply_grayscale, apply_canny, apply_color_quantization, apply_custom_color_quantization, apply_find_contours)

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
        else:
            processed_images = [image]

        # Convert the processed images back to PNG format and generate the HTML output
        output = ""
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
    </select>
    <input type="number" name="num_colors" min="1" max="10" value="3" id="numColorsQuant" style="display:none;" required>

    <input type="number" name="num_colors" min="1" max="10" value="3" id="numColors" style="display:none;" required>
    <div id="colorPickers" style="display:none;"></div>
    <button type="submit">Submit</button>
</form>
<script src="/static/main.js"></script>
''' + output



if __name__ == "__main__":
    app.run(debug=True)
