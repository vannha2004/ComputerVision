from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép truy cập từ frontend

# Load model và class indices
model = tf.keras.models.load_model("D:/KiVI/XLA/Product/OurProject/model/firstest.h5")
with open("D:/KiVI/XLA/Product/OurProject/model/class_indices.json") as f:
    class_indices = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def index():
    return "Flask server is running! Try POSTing to /predict"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    image_file = request.files['file']
    img_bytes = image_file.read()
    img_array = preprocess_image(img_bytes)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indices[str(predicted_index)]
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)