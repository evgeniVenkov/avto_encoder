import os
import torchvision.transforms as transforms, datasets

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from my_model import VAE as my_model

import numpy as np


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((200, 170)),
    transforms.ToTensor()
])

autoencoder = my_model().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()
scheduler = StepLR(optimizer, step_size=90, gamma=0.1)

image_dir = r'C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba'
images = os.listdir(image_dir)[:1000]

dataset = ImageDataset(image_dir, images, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

mass_loss = []


# Тренируем
def train():
    global mass_loss
    autoencoder.train()
    for epoch in range(100):

        for batch in dataloader:
            batch = batch.to(device)

            output, _, _ = autoencoder(batch)
            loss = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mass_loss.append(loss.item())

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch + 1}, Потери: {np.median(mass_loss):.4f} lr: {current_lr}")
        mass_loss = []



if __name__ == '__main__':
    train()
torch.save(autoencoder.state_dict(), "autoencoder_weights.pth")