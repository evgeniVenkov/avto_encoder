import os
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, 100, 85]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, 50, 43]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B, 128, 25, 21]
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 25 * 21, latent_dim)
        self.fc_logvar = nn.Linear(128 * 25 * 21, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 25 * 22)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [B, 64, 50, 44]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=(0, 1)),  # [B, 32, 100, 88]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding=(0, 1)),  # [B, 3, 200, 176]
            nn.Sigmoid()
        )

        # Кроп до точного размера 200x170
        self.crop = lambda x: x[:, :, :, :170]

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

        dec_input = self.decoder_input(latent).view(-1, 128, 25, 22)
        x_recon = self.decoder(dec_input)
        return self.crop(x_recon), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld * 0.001  # Вес KLD можно менять


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

transform = transforms.Compose([
    transforms.Resize((200, 170)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

faces = []
image_dir = r'C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba'
images = os.listdir(image_dir)

for img_name in tqdm(images[:50]):  # images — это список имён файлов
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    tensor_img = transform(img)
    faces.append(tensor_img)

faces = torch.stack(faces).to(device)

autoencoder = VAE()
autoencoder.load_state_dict(torch.load("autoencoder_weights.pth", map_location=device))
autoencoder.to(device)
autoencoder.eval()
out,latent,_ = autoencoder(faces)
reconstructed = out.detach().cpu()
show_faces(faces.cpu(), reconstructed)