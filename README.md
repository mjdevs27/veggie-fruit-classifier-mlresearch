AgriVision is a deep-learning powered image classification system that identifies 36 fruit and vegetable classes with high accuracy using a ResNet50 model trained on the Kaggle Fruit and Vegetable Image Recognition dataset.

This project includes:

✔️ A trained ResNet50 model

✔️ A clean Streamlit web app (local + cloud deploy)

✔️ Simple file-upload prediction interface

✔️ Confidence scores & top-class probabilities

✔️ Ready-to-deploy GitHub + Streamlit Cloud setup

🚀 Live Features

Upload any fruit/vegetable image (JPG/PNG)

Instantly get:

🏷️ Predicted class

📈 Confidence score

🔥 Top-3 probabilities

Fully offline local operation

Cloud deployable via Streamlit Cloud

📊 Dataset

Source:
Kaggle – Fruit and Vegetable Image Recognition
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/

36 classes

~3,800 images

Well-balanced dataset

Train/Val/Test split applied in Kaggle Notebook

🧠 Model Details

Architecture

Base model: ResNet50 (ImageNet weights)

Custom head:

GlobalAveragePooling2D

Dense (ReLU)

Dropout (0.4)

Dense Softmax Output

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Regularization:

Data Augmentation

EarlyStopping

ReduceLROnPlateau

Multi-GPU compatible (MirroredStrategy in training notebook)

Saved Model:
agrivision_resnet_best.keras

Class List:
Stored in class_names.json.