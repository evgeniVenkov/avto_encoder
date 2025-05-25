import os
import torchvision.transforms as transforms

from PIL import Image
from my_model import VAE as my_model
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split,Dataset

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


# === Параметры ===
BATCH_SIZE = 16
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_EPOCHS = 10

# === Загрузка данных ===
image_dir = r'C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba'
images = os.listdir(image_dir)[:30000]

dataset = ImageDataset(image_dir, images, transform=transform)

train_len = int(len(dataset) * TRAIN_RATIO)
val_len = int(len(dataset) * VAL_RATIO)
test_len = len(dataset) - train_len - val_len

train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# === Модель и оптимизация ===
autoencoder = my_model().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

train_losses = []
val_losses = []

# === Обучение ===
def train():
    for epoch in range(NUM_EPOCHS):
        autoencoder.train()
        running_loss = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            batch = batch.to(device)

            output, _, _ = autoencoder(batch)
            loss = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = np.mean(running_loss)
        train_losses.append(avg_train_loss)

        # === Валидация ===
        autoencoder.eval()
        val_loss = []

        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                output, _, _ = autoencoder(val_batch)
                loss = loss_fn(output, val_batch)
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr}")
        scheduler.step()

# === Запуск тренировки ===


if __name__ == '__main__':
    train()
torch.save(autoencoder.state_dict(), "autoencoder_weights.pth")


# === График потерь ===
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
