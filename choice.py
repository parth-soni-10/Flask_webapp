import cv2
from flask import Flask, render_template, request
import base64
import io
import os
from PIL import Image, ImageOps
import numpy as np

from tensorflow import keras
classify_model = keras.models.load_model('model\conj.h5')
classify_class_names = open("model\conj_labels.txt", "r").readlines()

#image_shape = (120,120,3)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/process_camera', methods=['POST'])
def process_image():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video capture")
        exit()
    # Read the frame from the camera
    ret, frame = cap.read()
    # Check if frame was successfully read
    if not ret:
        print("Error reading frame")
        exit()
    # Save the frame as an image
    cv2.imwrite("captured_image.png", frame)
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    filename = "captured_image.png"
    image = Image.open(filename)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict using the classification model
    class_prediction = classify_model.predict(data)
    class_index = np.argmax(class_prediction)
    class_name = classify_class_names[class_index]
    class_confidence_score = class_prediction[0][class_index]
    # Print the result
    if class_index == 1:
        conclusion = "White Eye, Not Infected"
        print(conclusion)
        grade_res = "low"
    else:
        conclusion = "Red Eye, Infected"
        print(conclusion)
        gra_model = keras.models.load_model('model\grading_keras_model.h5')
        grading_class_names = open("model\grading_labels.txt", "r").readlines()

        # Predict using the grading model
        grading_prediction = gra_model.predict(data)
        grading_index = np.argmax(grading_prediction)
        grade_res = grading_class_names[grading_index].strip()
        grade_res=grade_res[2:]        

    print(grade_res)
    if grade_res == "low":
            suggestions = ['Apply a warm compress to your eyes for 5-10 minutes several times a day to reduce discomfort.',
                                'Clean your eyelids and lashes using a mild soap and water or a commercial eyelid cleaner.',
                                'Avoid touching your eyes and wash your hands often to prevent the spread of infection.']
    if grade_res == "mid":        
            suggestions = ['Use lubricating eye drops or artificial tears to relieve dryness and irritation.',
                                'Take over-the-counter pain relievers, such as ibuprofen or acetaminophen, to reduce pain and inflammation.',
                                'Avoid wearing contact lenses until the symptoms have resolved.',
                                'If your symptoms worsen or do not improve after a few days, see a healthcare provider.']
    if grade_res == "high":
            suggestions = ['See a healthcare provider immediately for treatment, as severe cases of conjunctivitis can cause permanent vision damage if left untreated.',
                                'Follow the prescribed treatment plan, which may include antibiotic or antiviral eye drops or ointments.',
                                'Avoid wearing contact lenses until the symptoms have resolved.',
                                'Wash your hands often and avoid touching your eyes to prevent the spread of infection.']
    
    # Encode image as base64 for display in HTML
    with open(filename, "rb") as f:
        img_data = f.read()
        encoded_img = base64.b64encode(img_data).decode('utf-8')

    # Render results in a new HTML page
    return render_template('results.html', img_data=encoded_img, conclusion=conclusion, grade_res=grade_res, suggestions=suggestions)


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/process_upload', methods=['POST'])
def process_upload():
    # Get uploaded file
    file = request.files['file']
    # Read file data
    img_bytes = file.read()
    # Convert bytes to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    #img = Image.open(filename)
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict using the classification model
    class_prediction = classify_model.predict(data)
    class_index = np.argmax(class_prediction)
    class_name = classify_class_names[class_index]
    class_confidence_score = class_prediction[0][class_index]
    # Print the result
    if class_index == 1:
        conclusion = "White Eye, Not Infected"
        print(conclusion)
        grade_res = "low"
    else:
        conclusion = "Red Eye, Infected"
        print(conclusion)
        gra_model = keras.models.load_model('model\grading_keras_model.h5')
        grading_class_names = open("model\grading_labels.txt", "r").readlines()

        # Predict using the grading model
        grading_prediction = gra_model.predict(data)
        grading_index = np.argmax(grading_prediction)
        grade_res = grading_class_names[grading_index].strip()
        grade_res=grade_res[2:]        

    print(grade_res)
    if grade_res == "low":
            suggestions = ['Apply a warm compress to your eyes for 5-10 minutes several times a day to reduce discomfort.',
                                'Clean your eyelids and lashes using a mild soap and water or a commercial eyelid cleaner.',
                                'Avoid touching your eyes and wash your hands often to prevent the spread of infection.']
    if grade_res == "mid":        
            suggestions = ['Use lubricating eye drops or artificial tears to relieve dryness and irritation.',
                                'Take over-the-counter pain relievers, such as ibuprofen or acetaminophen, to reduce pain and inflammation.',
                                'Avoid wearing contact lenses until the symptoms have resolved.',
                                'If your symptoms worsen or do not improve after a few days, see a healthcare provider.']
    if grade_res == "high":
            suggestions = ['See a healthcare provider immediately for treatment, as severe cases of conjunctivitis can cause permanent vision damage if left untreated.',
                                'Follow the prescribed treatment plan, which may include antibiotic or antiviral eye drops or ointments.',
                                'Avoid wearing contact lenses until the symptoms have resolved.',
                                'Wash your hands often and avoid touching your eyes to prevent the spread of infection.']

    # Save processed image to disk
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, 'processed.png')
    img.save(filename)
    
    # Encode image as base64 for display in HTML
    with open(filename, "rb") as f:
        img_data = f.read()
        encoded_img = base64.b64encode(img_data).decode('utf-8')
        
    # Render results in a new HTML page
    return render_template('results.html', img_data=encoded_img, conclusion=conclusion, grade_res=grade_res,suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)
