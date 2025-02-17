import os
import pandas as pd
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_SAVE_PATH = "model/xray_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(len(train_generator.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)

with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

print(f"Image Classification Model saved to {MODEL_SAVE_PATH}")
print(f"Class indices saved to {CLASS_INDICES_PATH}")
DIABETES_DATA_PATH = "dataset/diabetes.csv"
CKD_DATA_PATH = "dataset/ckd.csv"
DIABETES_MODEL_FILENAME = "model/diabetes_model.pkl"
CKD_MODEL_FILENAME = "model/ckd_model.pkl"
def train_save_rf_model(data_path, target_column, model_filename):
    try:
        data = pd.read_csv(data_path)
        print(f"\nDataset loaded successfully from {data_path}.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please ensure the file exists.")
        return
    print(f"\nFirst 5 rows of the dataset from {data_path}:")
    print(data.head())
    print(f"\nMissing Values in Dataset from {data_path}:")
    print(data.isnull().sum())
    data = data.dropna()
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        print(f"Available columns: {data.columns.tolist()}")
        return
    print("\nEncoding categorical features...")
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    print(f"Categorical columns encoded: {categorical_columns}")
    if data[target_column].dtype == "object":
        data[target_column] = data[target_column].astype("category").cat.codes
    X = data.drop(columns=target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy for {model_filename}: {accuracy * 100:.2f}%")
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    with open(model_filename, "wb") as file:
        pickle.dump(rf_model, file)

    print(f"Model saved to {model_filename}")
train_save_rf_model(DIABETES_DATA_PATH, "Outcome", DIABETES_MODEL_FILENAME)
train_save_rf_model(CKD_DATA_PATH, "classification", CKD_MODEL_FILENAME) 