# %%


import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import albumentations as A
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf
import segmentation_models as sm


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()

IMAGE_DIR = '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/pngs_061024'
MASK_DIR = '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/masks_061024'

BASE_DIR = '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/split_data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

BACKBONE = 'inceptionv3'
BATCH_SIZE = 8
CLASSES = ['object']
LR = 0.0001
EPOCHS = 200
SIZE = 512
n_classes = 1
activation = 'sigmoid'

for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'masks'), exist_ok=True)

images = sorted(os.listdir(IMAGE_DIR))
masks = sorted(os.listdir(MASK_DIR))

assert len(images) == len(masks), "The number of images and masks should be equal"

train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

def copy_files(file_list, source_dir, dest_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

copy_files(train_images, IMAGE_DIR, os.path.join(TRAIN_DIR, 'images'))
copy_files(train_masks, MASK_DIR, os.path.join(TRAIN_DIR, 'masks'))
copy_files(val_images, IMAGE_DIR, os.path.join(VAL_DIR, 'images'))
copy_files(val_masks, MASK_DIR, os.path.join(VAL_DIR, 'masks'))
copy_files(test_images, IMAGE_DIR, os.path.join(TEST_DIR, 'images'))
copy_files(test_masks, MASK_DIR, os.path.join(TEST_DIR, 'masks'))

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

class Dataset:
    
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        image = cv2.resize(image, (SIZE, SIZE))
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
class Dataloder(keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

dataset = Dataset(os.path.join(TRAIN_DIR, 'images'), os.path.join(TRAIN_DIR, 'masks'), augmentation=None, preprocessing=None)

image, mask = dataset[5]
visualize(image=image, mask=mask)

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.PadIfNeeded(min_height=SIZE, min_width=SIZE, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=SIZE, width=SIZE, always_apply=True),
        A.OneOf([A.CLAHE(p=1), A.RandomGamma(p=1)], p=0.9),
        A.OneOf([A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1)], p=0.9),
        A.OneOf([A.HueSaturationValue(p=1)], p=0.9),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    return A.Compose([A.PadIfNeeded(384, 480, border_mode=cv2.BORDER_CONSTANT, value=0)])

def get_preprocessing(preprocessing_fn):
    _transform = [A.Lambda(image=preprocessing_fn)]
    return A.Compose(_transform)

preprocess_input = sm.get_preprocessing(BACKBONE)


with strategy.scope():
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    optim = keras.optimizers.Adam(LR)

    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optim, total_loss, metrics)

train_dataset = Dataset(
    os.path.join(TRAIN_DIR, 'images'), 
    os.path.join(TRAIN_DIR, 'masks'), 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

valid_dataset = Dataset(
    os.path.join(VAL_DIR, 'images'), 
    os.path.join(VAL_DIR, 'masks'), 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

assert train_dataloader[0][0].shape == (BATCH_SIZE, SIZE, SIZE, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, SIZE, SIZE, 1)

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)


# %% -- testing --

test_dataset = Dataset(
    os.path.join(TEST_DIR, 'images'), 
    os.path.join(TEST_DIR, 'masks'), 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

model.load_weights('best_model.h5') 

scores = model.evaluate(test_dataloader)


print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()
    
    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask[..., 0].squeeze(),
        pr_mask=pr_mask[..., 0].squeeze(),
    )

