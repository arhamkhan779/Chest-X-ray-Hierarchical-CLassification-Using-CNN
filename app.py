import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image

# Load your model
Model = keras.models.load_model('model.keras')  # Update with your model path

# Define classes
Super_Class = ['Normal', 'Abnormal']
Subclasses = ['Viral Pneumonia', 'Bacterial Pneumonia', 'Corona Virus', 'Tuberculosis']

def predict_image(image_path, model):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    img = cv2.resize(image, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)

    # Make predictions
    y_pred_binary, y_pred_subclass = model.predict(img)

    y_pred_binary = (y_pred_binary > 0.5).astype(int).flatten()
    y_pred_subclass = np.argmax(y_pred_subclass, axis=1)

    return y_pred_binary[0], y_pred_subclass[0], image

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f4f8; /* Light background */
        color: #333; /* Dark text for contrast */
    }
    .stFileUploader {
        margin: 20px 0;
    }
    .title {
        text-align: center;
        font-size: 36px;
        margin: 20px 0;
        color: #4A90E2; /* Blue color for the title */
    }
    .output {
        font-size: 20px;
        text-align: center;
        margin: 20px 0;
        color: #D94F00; /* Orange color for output text */
    }
    .image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;  /* Adjust the size of the image */
        border: 5px solid #4A90E2; /* Blue border for image */
        border-radius: 10px; /* Rounded corners */
    }
    .stButton {
        background-color: #5cb85c; /* Green color for buttons */
        color: white;
        border-radius: 5px;
    }
    .stButton:hover {
        background-color: #4cae4c; /* Darker green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.markdown('<h1 class="title">Medical Image Classification</h1>', unsafe_allow_html=True)

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an array
    image = Image.open(uploaded_file)
    image_path = uploaded_file.name
    image.save(image_path)  # Save to a temporary file for processing

    # Predict using the model
    y_pred_binary, y_pred_subclass, original_image = predict_image(image_path, Model)

    # Show the image
    st.image(original_image, caption='Uploaded Image', use_column_width=False, width=300)  # Adjusted width

    # Output results
    if y_pred_binary == 0:
        output = f"Super Class Output: {Super_Class[y_pred_binary]} | Subclass Output: NONE"
    else:
        output = f"Super Class Output: {Super_Class[y_pred_binary]} | Subclass Output: {Subclasses[y_pred_subclass]}"

    st.markdown(f'<p class="output">{output}</p>', unsafe_allow_html=True)
