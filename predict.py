import cv2
import numpy as np
from skimage.feature import hog
import joblib
import argparse

def predict_single_image(image_path):
    # Load trained model and scaler
    model = joblib.load('svm_burger_taco_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    
    # Resize to 128x128 (same as training)
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert to grayscale (for HOG)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    
    # Reshape and scale features
    hog_features = hog_features.reshape(1, -1)
    scaled_features = scaler.transform(hog_features)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    
    # Map prediction to class name
    class_names = ['burger', 'taco']
    return class_names[prediction]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict image class: burger or taco.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image.')
    args = parser.parse_args()

    try:
        result = predict_single_image(args.image_path)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")