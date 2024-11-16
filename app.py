from flask import Flask, request, send_file, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained Swift-SRGAN model
model = load_model('swift_srgan_model.h5')

# Preprocess input image for the model
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)

# Enhance the image using the model
def enhance_image(model, input_image):
    enhanced = model.predict(input_image)  # Get enhanced image from model
    enhanced = (enhanced[0] * 255).astype('uint8')  # Denormalize pixel values
    return Image.fromarray(enhanced)

@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        # Check if an image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the uploaded image
        image_file = request.files['image']
        image = Image.open(image_file)

        # Preprocess and enhance the image
        input_image = preprocess_image(image)
        enhanced_image = enhance_image(model, input_image)

        # Save the enhanced image temporarily
        output_path = 'enhanced_image.png'
        enhanced_image.save(output_path)

        # Return the enhanced image as response
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on the default port
    app.run(host='0.0.0.0', port=5000)
      
