# Advanced-Text-Recognition-with-Convolutional-Recurrent-Neural-Network-Integration
Advanced Text Recognition with Convolutional Recurrent Neural Network Integration
# Text Detection based on Convolutional Recurrent Neural Network (CRNN)

This repository contains the code for the CRNN implementation on the [TRSynth100K Dataset](https://www.kaggle.com/eabdul/textimageocr) from [Kaggle](https://www.kaggle.com/).

### Installation

To install the dependencies run:
```
pip install -r requirements.txt
```

### TRSynth100K Dataset
The dataset contains 100k images containing texts in them. Each image is of size 40x160 pixels. The text in the images are given as labels. The goal is to identify the text in the image.

Download the data, unzip the file and place the folder ```TRSynth100K``` inside the ```data``` folder.

### Data processing
To process the data, run:
```
python preprocessing.py
```

### Train the model
To train the model, run:
```
python train.py
```

### Use the model to make predictions
To make predictions on an image, run:
```python predict.py ----test_img path_to_test_img```


![image](https://github.com/lokesh12190/Advanced-Text-Recognition-with-Convolutional-Recurrent-Neural-Network-Integration/assets/115156195/3fe7b281-d317-4018-a717-90c0a4deff13)


** objective:**
"The objective of this project is to enhance text detection capabilities in images, applicable in diverse areas such as captcha decoding and vehicle identification via license plates. Leveraging the synergy of Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequence analysis, this project introduces an innovative approach to text recognition using a Convolutional Recurrent Neural Network (CRNN) model."

**Data Description:**
"The project utilizes the comprehensive TRSynth100K dataset, featuring 100,000 labeled images. These images, primarily containing single-line text, serve as the basis for training and testing our CRNN model."

**Aim :**
"Our aim is to develop an advanced CRNN framework capable of accurately predicting text from single-line images, thereby setting a new standard in image-based text recognition."

**Tech Stack :**
- Programming Language: Python
- Key Libraries: OpenCV (cv2), PyTorch (torch), NumPy, Pandas, Albumentations, Matplotlib, Pickle

**Approach :**
1. Initial Setup: Import required libraries and download the TRSynth100K dataset.
2. Data Preprocessing: Address null values, create a comprehensive data frame, and establish character-to-integer mappings.
3. Model Training: Execute a rigorous training process, including data splitting, dataset creation, loss function definition, model initialization, optimizer setting, and training iterations.
4. Predictions: Employ the trained model for image text predictions, utilizing techniques like augmentations and softmax for accurate result
