function showColorPickers() {
    var processingMethod = document.querySelector('select[name="processing_method"]').value;
    var numColorsInput = document.getElementById('numColors');
    var colorPickersDiv = document.getElementById('colorPickers');
    
    if (processingMethod === 'custom_color_quantization') {
        numColorsInput.style.display = 'inline';
        colorPickersDiv.style.display = 'block';
        updateColorPickers();
    } else {
        numColorsInput.style.display = 'none';
        colorPickersDiv.style.display = 'none';
    }
}

function updateColorPickers() {
    var numColors = document.getElementById('numColors').value;
    var colorPickersDiv = document.getElementById('colorPickers');
    colorPickersDiv.innerHTML = '';
    for (var i = 0; i < numColors; i++) {
        colorPickersDiv.innerHTML += '<input type="color" name="color_' + i + '" required> ';
    }
}
function showNumColorsInput() {
    var processingMethod = document.querySelector('select[name="processing_method"]').value;
    var numColorsInput = document.getElementById('numColorsQuant');
    if (processingMethod === 'color_quantization') {
        numColorsInput.style.display = 'inline';
    } else {
        numColorsInput.style.display = 'none';
    }
}
function showSpacingSelector() {
    var processingMethod = document.querySelector('select[name="processing_method"]').value;
    var lineSpacingInput = document.getElementById('line_spacing');
    if (processingMethod === 'horizontal_line_filling') {
        lineSpacingInput.style.display = 'inline';
    } else {
        lineSpacingInput.style.display = 'none';
    }
}
function displayGcode(gcode) {
    const canvas = document.getElementById("gcodeCanvas");
    const ctx = canvas.getContext("2d");

    // Clear the canvas and set the scale
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // ctx.setTransform(1, 0, 0, 1, 0, canvas.height);

    const lines = gcode.split("\n");
    let x = 0;
    let y = 0;
    let penDown = false;

    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;

    for (const line of lines) {
        const tokens = line.split(" ");
        if (tokens[0] === "G0" || tokens[0] === "G1") {
            let newX = null;
            let newY = null;
            let z = null;

            for (const token of tokens) {
                if (token[0] === "X") {
                    newX = parseFloat(token.substring(1));
                } else if (token[0] === "Y") {
                    newY = parseFloat(token.substring(1));
                } else if (token[0] === "Z") {
                    z = parseFloat(token.substring(1));
                }
            }

            if (z !== null) {
                if (z === 0) {
                    penDown = false;
                } else {
                    penDown = true;
                }
            }

            if (penDown && newX !== null && newY !== null) {
                // console.log('Go from: ' + x + ',' +y +' to: '+ newX + ',' +newY )
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(newX, newY);
                ctx.stroke();
            }

            if (newX !== null) x = newX;
            if (newY !== null) y = newY;
        }
    }
}


document.querySelector('select[name="processing_method"]').addEventListener('change', showNumColorsInput);
showNumColorsInput();

document.querySelector('select[name="processing_method"]').addEventListener('change', showColorPickers);
document.getElementById('numColors').addEventListener('change', updateColorPickers);
showColorPickers();

document.querySelector('button[name="generate_gcode"]').addEventListener('click', (event) => {
    event.preventDefault();

    const formData = new FormData(document.querySelector('form'));

    // Set the correct submit_type value
    formData.set('submit_type', 'generate_gcode');

    // Send a POST request to the /generate_gcode endpoint
    fetch('/generate_gcode', {
        method: 'POST',
        body: formData
    })
        .then((response) => response.json())
        .then((data) => {
            const gcode = data.gcode;
            const gcodeDisplay = document.getElementById("gcode-display");

            // For textarea element
            gcodeDisplay.value = gcode;
            // Display the G-code animation
            displayGcode(data.gcode);

            // Display the contour images
            const contourImagesContainer = document.getElementById('contour-images');
            contourImagesContainer.innerHTML = '';

            const contourImageBase64 = data.contour_image;
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${contourImageBase64}`;
            contourImagesContainer.appendChild(img);

        });
});