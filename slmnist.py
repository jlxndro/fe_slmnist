import config
import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import datetime
import tensorflow as tf # type : ignore
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your model
model = load_model("slmnist4.h5")

# Set up Streamlit interface
st.title("Sign Language Alphabet Detection")
st.write("Upload an image of a sign language gesture to detect the corresponding alphabet.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
camera_input = st.camera_input("Capture an image of your gesture")
alphabet_mapping = [chr(i) for i in range(65, 91) if chr(i) not in ["J", "Z"]]
if camera_input is not None:
    # Open the captured image
    img = Image.open(camera_input).convert("L")  # Convert to grayscale
    st.image(img, caption="Captured Image.", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((28, 28))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f"Predicted Sign Language Alphabet: {alphabet_mapping[predicted_class]}")

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((28, 28))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f"Predicted Sign Language Alphabet: {alphabet_mapping[predicted_class]}")