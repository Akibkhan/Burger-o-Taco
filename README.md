# Burger-o-Taco
This project is a machine learning-based image classification task where the goal is to predict whether an image contains a burger or a taco. The system utilizes a Support Vector Machine (SVM) classifier combined with Histogram of Oriented Gradients (HOG) features, along with OpenCV for image preprocessing and feature extraction.

# Key Features:
- SVM Classification: Support Vector Machines are employed to train the model to distinguish between burgers and tacos based on extracted visual features.
- HOG Feature Extraction: Histogram of Oriented Gradients is used to extract relevant features from images, capturing edge and texture information that is useful for classification.
- OpenCV Preprocessing: OpenCV is used for image resizing, conversion to grayscale, and other preprocessing steps necessary for feature extraction and model input.
- Simple User Interface: Users can upload an image, and the system will predict whether it's a burger or taco.

# Dependencies:

- Python 3.x
- OpenCV
-Scikit-learn
-Numpy

# How It Works:

- Image Preprocessing: The input image is preprocessed using OpenCV, including resizing and grayscale conversion.

- HOG Feature Extraction: HOG features are extracted from the image, representing the local object appearance and shape of the object (burger or taco).

- Model Training: An SVM model is trained on a labeled dataset of burger and taco images.

- Prediction: Once trained, the model can predict the class (burger or taco) of new images.

# Steps to Run:

Clone the repository:

git clone https://github.com/akibkhan/burger-or-taco-svm.git


Install required libraries:

`` pip install -r requirements.txt ``
 

Run the prediction script:

`` python predict.py --image_path your_image.jpg ``

# Results:

The trained model can accurately predict whether an image contains a burger or a taco, achieving good performance with a high level of accuracy.
