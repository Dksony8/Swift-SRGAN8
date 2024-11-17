from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your model
model = torch.load('path_to_your_model.pth')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image = image.convert('RGB')

    # Preprocess the image (resize, normalize, etc.)
    image = preprocess_image(image)  # You need to implement this function

    # Enhance the image using the model
    enhanced_image = model(image)

    # Postprocess the image (convert back to PIL format, etc.)
    enhanced_image = postprocess_image(enhanced_image)  # You need to implement this function

    # Save the enhanced image to a buffer
    buffer = io.BytesIO()
    enhanced_image.save(buffer, format='JPEG')
    buffer.seek(0)

    return buffer.getvalue(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
