# %%


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, os.path.basename(img_path), os.path.basename(mask_path)


def visualize_predictions(generator, dataloader, device):
    generator.eval()
    images, masks, img_basenames, mask_basenames = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        masked_images = images * (1 - masks)
        generator_input = torch.cat((masked_images, masks), dim=1)
        generated_images = generator(generator_input, images, masks)
        generated_images = generated_images.cpu().detach()
    
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9))
    for i in range(6):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original Image\n{img_basenames[i]}')
        
        axes[1, i].imshow(masks[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Mask\n{mask_basenames[i]}')
        
        axes[2, i].imshow(generated_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2, i].axis('off')
        axes[2, i].set_title('Generated Image')

    plt.show()

def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        for i, (images, masks, _, _) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            
            masked_images = images * (1 - masks)
            generator_input = torch.cat((masked_images, masks), dim=1)

            
            optimizer_G.zero_grad()
            generated_images = generator(generator_input, images, masks)
            gen_loss = criterion(discriminator(generated_images), torch.ones_like(discriminator(generated_images)))
            gen_loss.backward()
            optimizer_G.step()

            
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(images), torch.ones_like(discriminator(images)))
            fake_loss = criterion(discriminator(generated_images.detach()), torch.zeros_like(discriminator(generated_images)))
            dis_loss = (real_loss + fake_loss) / 2
            dis_loss.backward()
            optimizer_D.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Gen Loss: {gen_loss.item()}, Dis Loss: {dis_loss.item()}")

        if epoch % 10 == 0:
            visualize_predictions(generator, dataloader, device)


batch_size = 16
learning_rate = 0.0002
num_epochs = 300
image_dir = "/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/pngs_061024"
mask_dir = "/mnt/shared/dils/projects/water_quality_temp/data/data_061024_combined/masks_061024"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, image_transform=image_transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, num_epochs, device)

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')







# %%
