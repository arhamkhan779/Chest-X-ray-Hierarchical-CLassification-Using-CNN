import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load your model
Model = keras.models.load_model('model.keras')  # Update with your model path

# Define classes
Super_Class = ['Normal', 'Abnormal']
Subclasses = ['Viral Pneumonia', 'Bacterial Pneumonia', 'Corona Virus', 'Tuberculosis']

def predict_image(image, model):
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Preprocess the image
    img = image.resize((64, 64))  # Resize image
    img = np.array(img) / 255.0  # Normalize pixel values

    # Check if the image has the correct number of channels
    if img.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    img = img.reshape(1, 64, 64, 3)  # Reshape for model input

    # Make predictions
    y_pred_binary, y_pred_subclass = model.predict(img)

    y_pred_binary = (y_pred_binary > 0.5).astype(int).flatten()
    y_pred_subclass = np.argmax(y_pred_subclass, axis=1)

    return y_pred_binary[0], y_pred_subclass[0]

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
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.markdown('<h1 class="title">Medical Image Classification</h1>', unsafe_allow_html=True)

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)

    # Predict using the model
    y_pred_binary, y_pred_subclass = predict_image(image, Model)

    # Show the image
    st.image(image, caption='Uploaded Image', use_column_width=False, width=300)  # Adjusted width

    # Output results
    if y_pred_binary == 0:
        output = f"Super Class Output: {Super_Class[y_pred_binary]} | Subclass Output: NONE"
    else:
        output = f"Super Class Output: {Super_Class[y_pred_binary]} | Subclass Output: {Subclasses[y_pred_subclass]}"

    st.markdown(f'<p class="output">{output}</p>', unsafe_allow_html=True)
