from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import logging
from werkzeug.utils import secure_filename


app = Flask(__name__)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
MODEL_DIR = "model"
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
XRAY_MODEL_PATH = os.path.join(MODEL_DIR, "xray_model.h5")
CKD_MODEL_PATH = os.path.join(MODEL_DIR, "ckd_model.pkl")


def load_model(path, model_type="pickle"):
    try:
        if model_type == "pickle":
            with open(path, "rb") as file:
                return pickle.load(file)
        elif model_type == "keras":
            return tf.keras.models.load_model(path)
    except FileNotFoundError:
        logging.error(f"Model file not found at: {path}")
        raise FileNotFoundError(f"Error: {path} not found. Ensure the model is trained and saved.")

# Load all models
try:
    diabetes_model = load_model(DIABETES_MODEL_PATH, model_type="pickle")
    xray_model = load_model(XRAY_MODEL_PATH, model_type="keras")
    ckd_model = load_model(CKD_MODEL_PATH, model_type="pickle")
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Class labels for X-ray predictions
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}

# Helper function to preprocess X-ray images
def preprocess_image(image_path):
    IMAGE_SIZE = (224, 224)
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize to [0, 1]
    return image

# Routes
@app.route("/")
def home():
    return render_template("index1.html")

# Diabetes prediction route
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        features = [
            request.form.get(key, type=float) for key in [
                "pregnancies", "glucose", "bloodpressure", "skinthickness",
                "insulin", "bmi", "dpf", "age"
            ]
        ]
        if None in features:
            raise ValueError("All fields are required and must be valid numbers.")

        features = np.array([features])
        prediction = diabetes_model.predict(features)[0]
        result = "Positive" if prediction == 1 else "Negative"
        return render_template("diabetes.html", diabetes_result=f"Diabetes Prediction: {result}")
    except Exception as e:
        return render_template("diabetes.html", diabetes_result=f"Error: {str(e)}")

# X-ray prediction route
@app.route("/xray")
def xray():
    return render_template("xray.html")

@app.route("/predict_xray", methods=["POST"])
def predict_xray():
    if "file" not in request.files:
        return render_template("xray.html", xray_result="No file uploaded!")

    file = request.files["file"]
    if file.filename == "":
        return render_template("xray.html", xray_result="No selected file!")

    file_path = os.path.join(STATIC_DIR, secure_filename(file.filename))
    file.save(file_path)

    try:
        image = preprocess_image(file_path)
        predictions = xray_model.predict(image)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]
        result = f"{CLASS_LABELS[class_index]} ({confidence * 100:.2f}%)"
        return render_template("xray.html", xray_result=f"X-ray Prediction: {result}")
    except Exception as e:
        return render_template("xray.html", xray_result=f"Error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# CKD prediction route
@app.route("/ckd")
def ckd():
    return render_template("ckd.html")

@app.route("/predict_ckd", methods=["POST"])
def predict_ckd():
    try:
        features = [
            request.form.get(key, type=float) for key in [
                "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bu", "sc"
            ]
        ]
        if None in features:
            raise ValueError("All fields are required and must be valid numbers.")

        features = np.array([features])
        prediction = ckd_model.predict(features)[0]
        result = "Positive" if prediction == 1 else "Negative"
        return render_template("ckd.html", ckd_result=f"CKD Prediction: {result}")
    except Exception as e:
        return render_template("ckd.html", ckd_result=f"Error: {str(e)}")

# Run the Flask app
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="127.0.0.1", port=5000, debug=True)

