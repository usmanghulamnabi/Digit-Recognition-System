import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image, ImageOps, ImageFilter
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from streamlit_drawable_canvas import st_canvas

MODEL_FILE = "mnist_mlp_final.pkl"

# ==============================
# TRAIN MODEL (first run only)
# ==============================
@st.cache_resource
def train_model():
    st.write("Training high-accuracy MLP model... this may take a few minutes")

    # Load full MNIST dataset
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    X = X / 255.0
    y = y.astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Multi-layer perceptron with 3 hidden layers
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=50,
        verbose=True
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_FILE)

    acc = model.score(X_test, y_test)
    st.write(f"Model trained! Test accuracy: {acc*100:.2f}%")
    return model

# Load model if exists, else train
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = train_model()

# ==============================
# IMAGE PREPROCESSING FUNCTION
# ==============================
def preprocess_image(img):
    """Crops, centers using center of mass, scales, and normalizes drawn digit"""
    img = img.convert('L')                    # grayscale
    img = ImageOps.invert(img)                # invert colors
    img = img.filter(ImageFilter.GaussianBlur(1))

    img_arr = np.array(img)

    # Crop to bounding box
    coords = np.column_stack(np.where(img_arr > 0))
    if coords.size == 0:
        return np.zeros((1, 28*28))  # empty input

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img_arr = img_arr[y0:y1+1, x0:x1+1]

    # Resize to 20x20 while keeping aspect ratio
    img_pil = Image.fromarray(img_arr)
    img_pil.thumbnail((20, 20), Image.Resampling.LANCZOS)
    img_arr = np.array(img_pil)

    # Center using center of mass
    new_img = np.zeros((28,28), dtype=np.uint8)
    h, w = img_arr.shape
    cy, cx = np.array(img_arr.nonzero()).mean(axis=1)
    top = int(14 - cy)
    left = int(14 - cx)
    # Ensure indices are within bounds
    top = max(0, min(28 - h, top))
    left = max(0, min(28 - w, left))
    new_img[top:top+h, left:left+w] = img_arr

    # Normalize and flatten
    new_img = new_img / 255.0
    return new_img.reshape(1, -1)

# ==============================
# STREAMLIT UI
# ==============================
st.title("🧠 High-Accuracy MNIST Digit Recognition")

st.write("Draw a digit (0–9) below:")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        processed = preprocess_image(img)
        prediction = model.predict(processed)
        st.success(f"Prediction: {prediction[0]}")