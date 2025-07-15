import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import requests
import hashlib

app = Flask(__name__)

# Configuration
IMG_SIZE = (224, 224)
class_names = ['Anthracnose', 'Dry_Leaf', 'Healthy', 'Leaf_Spot']

# Model configuration
MODEL_URL = "https://your-cloud-storage-url/rubber_leaf_model_best.h5"  # Replace with your URL
MODEL_PATH = "rubber_leaf_model_best.h5"
MODEL_CHECKSUM = "your-model-checksum"  # Optional: for integrity check

# Load model once when app starts
model = None
model_loaded = False
model_error = None

def download_model():
    """Download model from cloud storage"""
    try:
        if os.path.exists(MODEL_PATH):
            print("Model file already exists locally")
            return True
        
        print("Downloading model from cloud storage...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.1f}%")
        
        print("Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def load_model_once():
    global model, model_loaded, model_error
    if not model_loaded:
        try:
            # Download model if not exists
            if not os.path.exists(MODEL_PATH):
                if not download_model():
                    raise Exception("Failed to download model")
            
            print(f"Loading model from: {MODEL_PATH}")
            print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
            
            model = load_model(MODEL_PATH)
            model_loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            model_error = str(e)
            print(f"Error loading model: {e}")

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
        "model_loaded": model_loaded,
        "model_error": model_error,
        "classes": class_names,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / (1024*1024), 2) if os.path.exists(MODEL_PATH) else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict rubber leaf disease from uploaded image"""
    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                "error": f"Model not loaded. Error: {model_error}",
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
        if not model_loaded:
            return jsonify({
                "error": f"Model not loaded. Error: {model_error}",
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

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload the model"""
    global model, model_loaded, model_error
    model = None
    model_loaded = False
    model_error = None
    
    # Remove existing model file to force re-download
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    
    load_model_once()
    
    return jsonify({
        "message": "Model reload attempted",
        "model_loaded": model_loaded,
        "model_error": model_error
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Rubber Leaf Disease Classification API",
        "model_classes": class_names,
        "model_status": {
            "loaded": model_loaded,
            "error": model_error
        },
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Upload image file for prediction",
            "/predict_base64": "POST - Send base64 encoded image for prediction",
            "/reload_model": "POST - Reload model"
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
