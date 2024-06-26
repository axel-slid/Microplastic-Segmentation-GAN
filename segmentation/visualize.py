# %%
import os
import numpy as np
import cv2
from PIL import Image
from openpyxl.drawing.image import Image as XLImage
from openpyxl import Workbook
from torchvision import transforms
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf
import segmentation_models as sm
import albumentations as A

# Set the environment variable for visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set model configuration parameters
BACKBONE = 'efficientnetb3'
CLASSES = ['object']
n_classes = 1
activation = 'sigmoid'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Load the pre-trained model with the specified backbone and weights
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights('/mnt/shared/dils/projects/water_quality_temp/code/model/best_model.h5')

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

# Function to predict the mask for a given image
def predict_image(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (SIZE, SIZE))
    
    preprocess_input = sm.get_preprocessing(BACKBONE)
    preprocessing = get_preprocessing(preprocess_input)
    sample = preprocessing(image=image)
    image = sample['image']
    
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()
    
    return denormalize(image.squeeze()), pr_mask.squeeze()

# Set directories for images and masks
IMAGE_DIR = '/mnt/shared/dils/projects/water_quality_temp/data/png_052924'
MASK_DIR = '/mnt/shared/dils/projects/water_quality_temp/data/masks_052924'

# Load and sort image and mask file names
image_files = sorted(os.listdir(IMAGE_DIR))[25:]
mask_files = sorted(os.listdir(MASK_DIR))[25:]

# Ensure the number of images matches the number of masks
assert len(image_files) == len(mask_files), "The number of images and masks do not match."

# Create a new workbook and get the active worksheet
workbook = Workbook()
worksheet = workbook.active

# Define transformations for the images
transformations = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.CenterCrop(SIZE)
])

# Loop through each image file and process
for index, image_file in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, image_file)
    mask_file = mask_files[index]
    mask_path = os.path.join(MASK_DIR, mask_file)
    
    print(f"Processing {index + 1}/{len(image_files)}: {image_file}")

    # Load and transform the image and mask
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img = transformations(img)
    mask = transformations(mask)

    # Save transformed images
    img.save('/tmp/modified_img' + str(index) + '.png')
    mask.save('/tmp/modified_img_mask' + str(index) + '.png')

    # Predict the mask using the model
    _, pr_mask = predict_image(model, img_path)
    pr_mask_img = Image.fromarray((pr_mask * 255).astype(np.uint8))
    pr_mask_img.save('/tmp/modified_img_pred' + str(index) + '.png')

    # Load images and corresponding masks into the Excel worksheet
    img = XLImage('/tmp/modified_img' + str(index) + '.png')
    mask = XLImage('/tmp/modified_img_mask' + str(index) + '.png')
    pr_mask_img = XLImage('/tmp/modified_img_pred' + str(index) + '.png')

    img_width, img_height = img.width, img.height
    worksheet.row_dimensions[index + 1].height = img_height
    worksheet.column_dimensions['A'].width = 40
    worksheet.add_image(img, 'A{}'.format(index + 1))
    worksheet.add_image(mask, 'B{}'.format(index + 1))
    worksheet.add_image(pr_mask_img, 'C{}'.format(index + 1))
    worksheet.cell(row=index + 1, column=4, value=img_path)
    
# Save the workbook
workbook.save('/mnt/shared/dils/projects/water_quality_temp/Vis.xlsx')
