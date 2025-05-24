import os

import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm



import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # до последнего conv
        # output: [B, 512, 7, 6] для входа 200x170

    def forward(self, x):
        return self.encoder(x)  # [B, 512, 7, 6]

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> [B, 256, 14, 12]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [B, 128, 28, 24]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> [B, 64, 56, 48]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> [B, 32, 112, 96]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # -> [B, 16, 224, 192]
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),                         # -> [B, 3, 224, 192]
            nn.Sigmoid()
        )
        self.crop = nn.Identity()  # заменим позже

    def forward(self, x):
        x = self.decoder(x)
        # обрезаем до [B, 3, 200, 170]
        return x[:, :, :200, :170]  # simple crop

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)  # [B, 512, 7, 6]
        out = self.decoder(z)
        return out  # [B, 3, 200, 170]

def denormalize(tensor):
    return (tensor + 1) / 2  # из [-1, 1] → [0, 1]


def show_faces(original, reconstructed, n=5):
    # original = denormalize(original)
    # reconstructed = denormalize(reconstructed)

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

image_dir = r'C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((200, 170)),
    transforms.ToTensor()
])

# mtcnn = MTCNN(image_size=160, margin=0, device=device)
autoencoder = Autoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Выбираем первые N лиц
N = 50
images = os.listdir(image_dir)[:N]
faces = []

for img_name in tqdm(images):  # images — это список имён файлов
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    tensor_img = transform(img)
    faces.append(tensor_img)

faces = torch.stack(faces).to(device)


# Тренируем автоэнкодер
for epoch in range(100):
    autoencoder.train()
    output = autoencoder(faces)
    loss = loss_fn(output, faces)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Эпоха {epoch+1}, Потери: {loss.item():.4f}")

# Показываем результат
autoencoder.eval()
reconstructed = autoencoder(faces).detach().cpu()

show_faces(faces.cpu(), reconstructed)
