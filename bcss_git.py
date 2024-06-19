import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path

RESIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

LABEL_MAP = {
    0: "outside_roi",
    1: "tumor",
    2: "stroma",
    3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris",
    5: "glandular_secretions",
    6: "blood",
    7: "exclude",
    8: "metaplasia_NOS",
    9: "fat",
    10: "plasma_cells",
    11: "other_immune_infiltrate",
    12: "mucoid_material",
    13: "normal_acinus_or_duct",
    14: "lymphatics",
    15: "undetermined",
    16: "nerve",
    17: "skin_adnexa",
    18: "blood_vessel",
    19: "angioinvasion",
    20: "dcis",
    21: "other",
}

preprocess_input = smp.encoders.get_preprocessing_fn(
    encoder_name="resnet34", pretrained="imagenet"
)

image_transforms = transforms.Compose(
    [
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

mask_transforms = transforms.Compose([transforms.Resize(RESIZE), transforms.ToTensor()])

# 1. Streamlit Interface Setup
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Welcome to my awesome data science project!")
    st.text("In this project, we perform breast cancer semantic segmentation.")

with dataset:
    st.header("Dataset Overview")
    st.text("This app allows you to upload an image for segmentation.")

with features:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with model_training:
    st.header("Model Prediction")
    
    if uploaded_file is not None:
        # 3. Model Loading & Prediction
        @st.cache_resource
        def load_model():
            model = smp.PSPNet(in_channels=3, classes=21)
            model_path = "/Users/dhyeydesai/Desktop/ML_DL_Projects/PUBLICATIONS/BCSS/BREAST_CANCER_SEMANTIC_SEGMENTATION/best_model.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
            model.eval()
            return model

        model = load_model()

        # 4. Preprocessing and Inference
        image_tensor = image_transforms(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probability_map = output.softmax(dim=1)
            prediction_mask = probability_map.argmax(dim=1).squeeze().cpu()

        # 5. Dominant Class Determination
        unique_classes, class_counts = np.unique(prediction_mask.numpy(), return_counts=True)
        dominant_class_index = unique_classes[np.argmax(class_counts)]
        dominant_class_label = LABEL_MAP.get(dominant_class_index, "Unknown")  # Handle unknown classes

        # 6. Visualization and Class Display
        prediction_mask_np = prediction_mask.numpy()
        prediction_mask_color = np.zeros((*prediction_mask_np.shape, 3), dtype=np.uint8)
        for i in range(21):
            prediction_mask_color[prediction_mask_np == i] = (
                np.array(plt.get_cmap("tab20")(i)[:3]) * 255
            )

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(prediction_mask_color)
        ax[1].set_title(f"Predicted Segmentation\nDominant Class: {dominant_class_label}")
        ax[1].axis("off")

        st.pyplot(fig)
