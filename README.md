# skincancerdetect
Skin Cancer Classification using Vision Transformer (ViT)
This project implements a deep learning pipeline using the Vision Transformer (ViT) model to classify different types of skin cancer using the HAM10000 dataset.

Table of Contents
Dataset
Installation
Pipeline
Results
Usage
Acknowledgements
Dataset
The HAM10000 dataset is used for this project. It contains 10,015 dermatoscopic images of skin lesions classified into 7 categories:

akiec: Actinic keratoses and intraepithelial carcinoma
bcc: Basal cell carcinoma
bkl: Benign keratosis-like lesions
df: Dermatofibroma
mel: Melanoma
nv: Melanocytic nevi
vasc: Vascular lesions
Installation
Prerequisites
Python >= 3.8
PyTorch >= 1.10
Hugging Face Transformers
Steps
Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/vit-skin-cancer-classification.git
cd vit-skin-cancer-classification
Install dependencies:

bash
Copy code
pip install transformers datasets torch torchvision kaggle matplotlib seaborn scikit-learn pandas
Download and unzip the dataset:

bash
Copy code
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip
Pipeline
Preprocessing
Data Balancing: Resampled each class to 500 samples for balanced training.
Data Augmentation: Images resized to 224x224, and preprocessing applied using ViT's feature extractor.
Dataset Split: Data divided into 80% training and 20% testing.
Model
The Vision Transformer (ViT) model vit-base-patch16-224-in21k from Hugging Face was fine-tuned for the 7-class classification task.

Training
Learning Rate: 2e-5
Batch Size: 16
Epochs: 5
Optimizer: AdamW
Evaluation Metric: Weighted Accuracy, Precision, Recall, F1-score
Results
Accuracy: 87.2%
F1-Score: 0.86
Precision: 0.88
Recall: 0.87
Confusion Matrix
The confusion matrix visualizes the model's performance across all classes.

Usage
To train the model, run:

bash
Copy code
python train.py
To evaluate the model:

bash
Copy code
python evaluate.py
Predictions can be made using:

python
Copy code
from transformers import ViTForImageClassification
from PIL import Image

model = ViTForImageClassification.from_pretrained('./vit-skin-cancer-classifier')
image = Image.open("path_to_image.jpg").convert("RGB")
# Preprocess image and pass to the model for prediction
Acknowledgements
Hugging Face for the Vision Transformer implementation
HAM10000 Dataset contributors
Kaggle community for dataset hosting
