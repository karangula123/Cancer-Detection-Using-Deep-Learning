import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template , jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing import image



app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
# ------------------------------------------blood cancer detection--------------------------------------------

# Define your model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

blood_model_path = 'model/cancer_classification_model.h5'

# Load the trained model weights
blood_model = tf.keras.models.load_model(blood_model_path)


@app.route('/blood_cancer', methods=['GET', 'POST'])
def blood_cancer():
    return render_template('blood_cancer.html')

 
def get_className(prediction):
    if prediction > 0.5:
        return "normal"
    else:
        return "cancer"

def getResult(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict using the model
    prediction = blood_model.predict(img_array)

    return prediction[0][0]  # return the prediction value

@app.route('/predict', methods=['POST'])
def blood_cancer_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        f = request.files['file']
        
        if f.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        os.makedirs(os.path.join(basepath, 'uploads'), exist_ok=True)  # Ensure the directory exists
        f.save(file_path)

        try:
            prediction = getResult(file_path)
            results = get_className(prediction)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Optionally, delete the file after processing
            if os.path.exists(file_path):
                os.remove(file_path)

        return jsonify({'prediction': results})
    
    return jsonify({'error': 'Invalid request method'}), 405


# --------------------------------------------------end of blood cancer---------------------------------------------

#---------------------lung cancer detection-------------------------------- 

# Define the model
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: adenocarcinoma, benign, squamous_carcinoma
])


lung_model_path = 'model/lung_cancer_model.h5'

# Load the trained model weights
lung_model = tf.keras.models.load_model(lung_model_path)

@app.route('/lung_cancer', methods=['GET', 'POST'])
def lung_cancer():
    return render_template('lung_cancer.html')

 #Function to preprocess the image

# Function to preprocess the image for lung cancer model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to get the class name from lung cancer prediction
def get_class_name(prediction):
    class_labels = {0: 'adenocarcinoma', 1: 'benign', 2: 'squamous_carcinoma'}
    predicted_class = np.argmax(prediction, axis=1)
    return class_labels[predicted_class[0]]

# Route for uploading the lung cancer image and making predictions
@app.route('/lung_cancer_predict', methods=['POST'])
def lung_cancer_predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        os.makedirs(os.path.join(basepath, 'uploads'), exist_ok=True)  # Ensure the directory exists
        f.save(file_path)
        
        # Preprocess the image for lung cancer model
        img_array = preprocess_image(file_path)
        
        # Predict using the lung cancer model
        prediction = lung_model.predict(img_array)
        
        # Get the class name from lung cancer prediction
        class_name = get_class_name(prediction)
        
        return jsonify({'class_name': class_name})
    
    return jsonify({'error': 'Prediction Failed'})



# -------------------------------------------skin cancer model--------------------------------------
# Define your model
model = Sequential([
     Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

skin_model_path = 'model/skin_cancer_classifier.h5'

# Load the trained model weights
skin_model = tf.keras.models.load_model(skin_model_path)

@app.route('/skin_cancer', methods=['GET', 'POST'])
def skin_cancer():
    return render_template('skin_cancer.html')

# Function to preprocess the image
def preprocess_image(img_path):
    """Preprocess the image to the required format for the model."""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict the class of the image
def predict_image_class(img_path):
    """Make a prediction and return the class."""
    img_array = preprocess_image(img_path)
    prediction = skin_model.predict(img_array)
    if prediction[0] > 0.5:
        return "Malignant"
    else:
        return "Benign"

# Route for handling image uploads and predictions
@app.route('/skin_cancer_predict', methods=['POST'])
def skin_upload():
    if request.method == 'POST':
        # Check if the POST request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        # Check if the file name is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        
        # Save the file to a specified location
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
        file_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(file_path)
        
        # Perform prediction on the uploaded image
        prediction = predict_image_class(file_path)
        
        # Remove the temporary file
        os.remove(file_path)
        
        # Return the prediction result as JSON response
        return jsonify({'prediction': prediction})
    
    return jsonify({'error': 'Invalid request method'}), 405




# @app.route('/skin_cancer', methods=['GET', 'POST'])
# def skin_cancer():
#     return render_template('skin_cancer.html')

if __name__ == '__main__':
    app.run(debug=True)