import csv
import os

import numpy as np
import tensorflow as tf
from PIL import Image

# from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the saved gender model
model_path_gender = "app/model/V4_30-AAF+UTK _selected_Gender_Prediction.h5"
model_g = tf.keras.models.load_model(model_path_gender)

# Load the saved age model
model_path_age = "app/model/V4_30-AAF+UTK_Age_Prediction.h5"
model_a = tf.keras.models.load_model(model_path_age, compile=False)


def genderAge(pic_path, image):
    # Function to preprocess the image
    # gender
    def preprocess_image_gender(image):
        img = Image.open(os.path.join(pic_path, image)).convert(
            "L"
        )  # Convert to grayscale
        img = img.resize(
            (128, 128), Image.LANCZOS
        )  # Use LANCZOS for high-quality downsampling
        img = np.array(img)
        img = img.reshape(1, 128, 128, 1)
        img = img / 255.0
        return tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to tf.Tensor

    # age
    def preprocess_image_age(image):
        img = Image.open(os.path.join(pic_path, image)).convert("RGB")  # Convert to RGB
        img = img.resize(
            (128, 128), Image.LANCZOS
        )  # Use LANCZOS for high-quality downsampling
        img = np.array(img)
        img = img.reshape(1, 128, 128, 3)
        img = img / 255.0
        return tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to tf.Tensor

    # Function to predict gender and age
    def predict_gender_age(image):
        # gender
        img_g = preprocess_image_gender(image)
        predictions_g = model_g(inputs=img_g)
        gender_prediction = predictions_g[0][0]  # Update access to the predicted output
        gender = "Male" if gender_prediction < 0.5 else "Female"

        # age
        img_a = preprocess_image_age(image)
        predictions_a = model_a(inputs=img_a)
        age_prediction = predictions_a[0][0]  # Update access to the predicted output
        age = int(age_prediction)

        return gender, age

    gender, age = predict_gender_age(image)
    print("Image:", image)
    print("Predicted Gender:", gender)
    print("Predicted Age:", age)
    print("-----------------------")

    # Create a CSV file to store the results
    csv_file = "app/data/gender_age_predictions.csv"
    column_names = ["Image", "Predicted Gender", "Predicted Age"]

    if not os.path.exists(csv_file):
        with open(csv_file, "a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(column_names)

    # Write the predictions to the CSV file
    with open(csv_file, "a", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([image, gender, age])
