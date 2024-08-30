##Breast Cancer Semantic Segmentation

Welcome to my Data Science Project! This project focuses on performing breast cancer semantic segmentation using a deep learning model. The application is built with Streamlit, allowing users to upload images and receive segmented outputs indicating different classes related to breast cancer tissues.

## Table of Contents
- [App Overview](#app-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Preprocessing](#preprocessing)
- [Classes](#classes)
- [Dataset](#dataset)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## App Overview

This app allows users to upload an image for segmentation. The application will process the image using a pre-trained deep learning model and provide a segmented output, highlighting different tissue types relevant to breast cancer analysis.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   
   git clone https://github.com/DHYEY166/BREAST_CANCER_SEMANTIC_SEGMENTATION.git
   
   cd breast-cancer-segmentation

3. Create a virtual environment (optional but recommended):

   python -m venv venv
   
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

5. Install the required packages:

   pip install -r requirements.txt

6. Run the Streamlit application:

   streamlit run bcss_git.py

## Usage

- **Upload an Image**: You can upload an image in PNG, JPG, or JPEG format through the application interface.
- **Model Prediction**: Once the image is uploaded, the model will process it and display the segmented output.
- **Segmentation Results**: The application will show the original image alongside the segmentation results, indicating the dominant class in the image.

You can also access the application directly via the following link:

[Streamlit Application](https://breastcancersemanticsegmentation-4ei9goqle5y39zzmejl3pm.streamlit.app)

## Model Details

The application utilizes the PSPNet architecture implemented using the segmentation_models_pytorch library. The model was trained on the Breast Cancer Semantic Segmentation (BCSS) dataset.

## Preprocessing

- **Image Resize**: All images are resized to 224x224 pixels.
- **Normalization**: The images are normalized using the mean and standard deviation values from the ImageNet dataset.

## Classes

The model predicts 21 classes, including tumor, stroma, lymphocytic infiltrate, and others. Each class is mapped to a unique label for easier interpretation of the segmented output.

## Dataset

The model was trained using the [Breast Cancer Semantic Segmentation (BCSS)](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss) dataset available on Kaggle. The dataset contains images with annotations of different breast cancer tissue types.

## Features

- **Image Upload**: Users can upload images in PNG, JPG, or JPEG formats.
- **Model Prediction**: The model segments the image and identifies the dominant class present in the tissue.
- **Visualization**: The original image and the segmented output are displayed side by side for easy comparison.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/DHYEY166/BREAST_CANCER_SEMANTIC_SEGMENTATION/blob/main/LICENSE) file for more details.

## Contact

- **Author**: Dhyey Desai
- **Email**: dhyeydes@usc.edu
- **GitHub**: https://github.com/DHYEY166
- **LinkedIn**: https://www.linkedin.com/in/dhyey-desai-80659a216 

Feel free to reach out if you have any questions or suggestions.
