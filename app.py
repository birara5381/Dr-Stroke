from flask import Flask, request, render_template, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = load_model('efficientnet_b0_model.h5', compile=False)

# Class names (adjust as per your label encoder)
class_names = ['Hemorrhagic Stroke', 'Ischemic Stroke']  # Replace with your actual class names

def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        image = Image.open(file)
        prepared_image = prepare_image(image)
        prediction = model.predict(prepared_image)
        predicted_class = class_names[np.argmax(prediction)]
        
        # Save the image with the predicted class name
        filename = f"{predicted_class}_{file.filename}"
        save_path = os.path.join('static/uploads', filename)
        image.save(save_path)

        return render_template('index.html', prediction=predicted_class, filename=filename)
    return "Error processing file"

if __name__ == '__main__':
    # Ensure the static/uploads directory exists
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
