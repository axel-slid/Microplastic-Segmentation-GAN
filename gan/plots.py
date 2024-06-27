# %%
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the Generator class, a neural network model for image generation
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the encoder part of the generator network
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # Define the decoder part of the generator network        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  
        )
    # Define the forward pass of the generator
    def forward(self, x, original_image, mask):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x * mask + original_image * (1 - mask)
        return x

# Load a pre-trained generator model
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)

# Function to preprocess input images and masks
def preprocess(image_path, mask_path, image_transform, mask_transform):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    image = image_transform(image)
    mask = mask_transform(mask)

    return image, mask

# Function to perform inference using the generator model
def inference(image_path, mask_path, generator, device):
    # Define the transformations for the input image and mask
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Preprocess the input image and mask
    image, mask = preprocess(image_path, mask_path, image_transform, mask_transform)
    image = image.to(device).unsqueeze(0)
    mask = mask.to(device).unsqueeze(0)

    # Perform inference with the generator model
    with torch.no_grad():
        masked_image = image * (1 - mask)
        generator_input = torch.cat((masked_image, mask), dim=1)
        generated_image = generator(generator_input, image, mask)
        generated_image = generated_image.squeeze(0).cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5

    return generated_image

# Define a list of paths to input images and masks
image_path = '/mnt/shared/dils/projects/water_quality_temp/data/data_nonmicroplastic/non_microplastic/0011.png'
mask_path = '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/masks_061024/0333.png'

# Perform inference to generate an image
generated_image = inference(image_path, mask_path, generator, device)

# Plot the original image, mask, and generated image
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(Image.open(image_path))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Image.open(mask_path), cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(generated_image)
plt.title('Generated Image')
plt.axis('off')

plt.show()


# %%
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), 
        )

    def forward(self, x, original_image, mask):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x * mask + original_image * (1 - mask)
        return x


generator = Generator()
generator.load_state_dict(torch.load('generator.pth', map_location='cpu'))
generator.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)


def preprocess(image_path, mask_path, image_transform, mask_transform):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    image = image_transform(image)
    mask = mask_transform(mask)

    return image, mask


def inference(image_path, mask_path, generator, device):
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image, mask = preprocess(image_path, mask_path, image_transform, mask_transform)
    image = image.to(device).unsqueeze(0)
    mask = mask.to(device).unsqueeze(0)

    with torch.no_grad():
        masked_image = image * (1 - mask)
        generator_input = torch.cat((masked_image, mask), dim=1)
        generated_image = generator(generator_input, image, mask)
        generated_image = generated_image.squeeze(0).cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5

    return image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5, mask.cpu().squeeze(0).numpy(), generated_image


image_paths = [
    '/mnt/shared/dils/projects/water_quality_temp/data/data_nonmicroplastic/non_microplastic/0011.png',
    '/mnt/shared/dils/projects/water_quality_temp/data/data_nonmicroplastic/non_microplastic/0013.png',
    '/mnt/shared/dils/projects/water_quality_temp/data/data_nonmicroplastic/non_microplastic/0211.png'
]
mask_paths = [
    '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/masks_061024/0333.png',
    '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/masks_061024/0320.png',
    '/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/masks_061024/0334.png'
]


fig, axs = plt.subplots(3, 3, figsize=(12, 12))


fig.suptitle('      Original Image         Guiding Mask          Generated Image', fontsize=20, y=0.92)

for i in range(3):
    image_path = image_paths[i]
    mask_path = mask_paths[i]

    original_image, mask, generated_image = inference(image_path, mask_path, generator, device)

    axs[i, 0].imshow(original_image)
    axs[i, 0].axis('off')

    axs[i, 1].imshow(mask.squeeze(), cmap='gray')
    axs[i, 1].axis('off')

    axs[i, 2].imshow(generated_image)
    axs[i, 2].axis('off')


plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()


