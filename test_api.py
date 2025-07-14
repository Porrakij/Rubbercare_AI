import requests
import base64
import json

# Test script for the deployed API
API_URL = "http://localhost:5000"  # Change this to your Railway URL when deployed

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_file_upload(image_path):
    """Test file upload prediction"""
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print("File Upload Prediction:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_base64_prediction(image_path):
    """Test base64 prediction"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    data = {'image': image_base64}
    response = requests.post(f"{API_URL}/predict_base64", json=data)
    
    print("Base64 Prediction:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

if __name__ == "__main__":
    # Test health endpoint
    test_health()
    
    # Test with an image (replace with your test image path)
    test_image_path = "test_image.jpg"  # Replace with actual image path
    
    try:
        test_file_upload(test_image_path)
        test_base64_prediction(test_image_path)
    except FileNotFoundError:
        print(f"Test image not found: {test_image_path}")
        print("Please provide a valid image path to test the prediction endpoints.")