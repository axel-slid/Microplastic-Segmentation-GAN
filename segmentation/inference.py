# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import albumentations as A
import tensorflow as tf
import segmentation_models as sm

# Set the environment variable for the segmentation models framework
os.environ['SM_FRAMEWORK'] = 'tf.keras'

# Function to load and preprocess an image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    return image

# Function to visualize and save images
def visualize(save_path, **images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(save_path)
    plt.show()

# Function to denormalize image data
def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

# Function to get preprocessing transformations
def get_preprocessing(preprocessing_fn):
    _transform = [A.Lambda(image=preprocessing_fn)]
    return A.Compose(_transform)

# Function to predict and visualize the mask for a given image
def predict_image(model, image_path, save_path):
    # Load and preprocess the image
    image = load_image(image_path)  
    preprocess_input = sm.get_preprocessing(BACKBONE)
    preprocessing = get_preprocessing(preprocess_input)
    sample = preprocessing(image=image)
    image = sample['image']
    # Expand dimensions and make prediction
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()

    # Visualize and save the input image and predicted mask
    visualize(
        save_path,
        image=denormalize(image.squeeze()),
        pr_mask=pr_mask.squeeze(),
    )


# Set the environment variable for visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set model configuration parameters
BACKBONE = 'efficientnetb3'
CLASSES = ['object']
n_classes = 1
activation = 'sigmoid'
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights('best_model.h5')

# Define the image path and save path for the prediction
image_path = '/mnt/shared/dils/projects/water_quality_temp/data/split_data/test/images/0071.png'
save_path = '/mnt/shared/dils/projects/water_quality_temp/code/model/predictions/img_pred.png'
predict_image(model, image_path, save_path)
