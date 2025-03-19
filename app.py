import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Load the trained model (Ensure the model is in your repository)
@st.cache_resource
def load_model():
    model = torch.load("models/best_unet_model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

# Image Preprocessing Function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = np.array(image)
    image = cv2.resize(image, (256, 256))  # Resize to model input size
    image = image / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    return image

# Segmentation Prediction
def predict_mask(model, image):
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        return output

# Streamlit UI
st.title("Breast Cancer Image Segmentation")
st.write("Upload a breast cancer image, and the model will generate the segmented mask.")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_model()
    
    # Preprocess Image
    input_image = preprocess_image(image)

    # Predict Mask
    predicted_mask = predict_mask(model, input_image)

    # Convert Mask to Displayable Format
    predicted_mask = (predicted_mask * 255).astype(np.uint8)

    # Display Result
    st.image(predicted_mask, caption="Predicted Segmentation Mask", use_column_width=True)
