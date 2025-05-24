import os
import torchvision.transforms as transforms, datasets
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

class ImageDataset(Dataset):
    def __init__(self, image_dir, images_list, transform=None):
        self.image_dir = image_dir
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = self.images_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # [B,64,100,85]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B,128,50,42]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # [B,256,25,21]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 25 * 21, latent_dim)
        self.fc_logvar = nn.Linear(256 * 25 * 21, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 25 * 21)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B,128,50,42]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=(0, 1)),  # [B,64,100,85]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, output_padding=0),  # [B,3,200,170]
            nn.Sigmoid()
        )

        # Кроп не нужен, т.к. размер совпадает
        self.crop = lambda x: x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z = self.encoder(x)
        z_flat = self.flatten(z)
        mu = self.fc_mu(z_flat)
        logvar = self.fc_logvar(z_flat)
        latent = self.reparameterize(mu, logvar)

        dec_input = self.decoder_input(latent).view(-1, 256, 25, 21)
        x_recon = self.decoder(dec_input)
        return self.crop(x_recon), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld * 1e-4  # Вес KLD можно регулировать

def show_faces(original, reconstructed, n=5):
    fig, axes = plt.subplots(2, n, figsize=(15, 5))
    for i in range(n):
        axes[0, i].imshow(original[i].permute(1, 2, 0).clip(0, 1))
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).clip(0, 1))
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Оригинал', fontsize=14)
    axes[1, 0].set_ylabel('После A.E.', fontsize=14)
    plt.tight_layout()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((200, 170)),
    transforms.ToTensor()
])

autoencoder = VAE().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()

image_dir = r'C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba'
images = os.listdir(image_dir)[:50000]

dataset = ImageDataset(image_dir, images, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

mass_loss = []


# Тренируем
def train():
    scheduler = StepLR(optimizer, step_size=900, gamma=0.1)

    for epoch in range(1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            batch = batch.to(device)
            autoencoder.train()
            output, _, _ = autoencoder(batch)
            loss = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mass_loss.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            scheduler.step()

        print(f"Эпоха {epoch + 1}, Потери: {np.median(mass_loss):.4f}")

    torch.save(autoencoder.state_dict(), "autoencoder_weights.pth")




if __name__ == '__main__':
    train()
