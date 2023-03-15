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

document.querySelector('select[name="processing_method"]').addEventListener('change', showNumColorsInput);
showNumColorsInput();

document.querySelector('select[name="processing_method"]').addEventListener('change', showColorPickers);
document.getElementById('numColors').addEventListener('change', updateColorPickers);
showColorPickers();
