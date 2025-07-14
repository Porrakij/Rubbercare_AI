import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configuration
IMG_SIZE = (224, 224)
class_names = ['Anthracnose', 'Dry_Leaf', 'Healthy', 'Leaf_Spot']

# Load model once when app starts
model = None

def load_model_once():
    global model
    if model is None:
        try:
            # Try to load from current directory first
            model_path = 'rubber_leaf_model_best.h5'
            if os.path.exists(model_path):
                model = load_model(model_path)
                print("Model loaded successfully from current directory")
            else:
                raise FileNotFoundError("Model file not found")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

def preprocess_image(image_data):
    """Preprocess image for prediction"""
    try:
        # Convert image data to PIL Image
        if isinstance(image_data, str):
            # If base64 string
            image_data = base64.b64decode(image_data)
        
        img = Image.open(io.BytesIO(image_data))
        img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict rubber leaf disease from uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "status": "error"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "status": "error"
            }), 400
        
        # Read and preprocess image
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = class_names[class_index]
        
        # Get all class probabilities
        all_predictions = {}
        for i, class_name in enumerate(class_names):
            all_predictions[class_name] = float(prediction[0][i])
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "all_predictions": all_predictions,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict from base64 encoded image"""
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "error": "No base64 image data provided",
                "status": "error"
            }), 400
        
        # Decode and preprocess image
        image_base64 = data['image']
        processed_image = preprocess_image(image_base64)
        
        # Make prediction
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = class_names[class_index]
        
        # Get all class probabilities
        all_predictions = {}
        for i, class_name in enumerate(class_names):
            all_predictions[class_name] = float(prediction[0][i])
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "all_predictions": all_predictions,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Rubber Leaf Disease Classification API",
        "model_classes": class_names,
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Upload image file for prediction",
            "/predict_base64": "POST - Send base64 encoded image for prediction"
        },
        "usage": {
            "file_upload": "Send POST request to /predict with 'image' file",
            "base64": "Send POST request to /predict_base64 with JSON: {'image': 'base64_string'}"
        }
    })

if __name__ == '__main__':
    # Load model on startup
    load_model_once()
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)