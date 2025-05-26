import os
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from my_model import VAE as my_model
import torch



def show_faces(original, reconstructed, n=6):

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

for img_name in images[30003:30009]:  # images — это список имён файлов
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    tensor_img = transform(img)
    faces.append(tensor_img)

faces = torch.stack(faces).to(device)

autoencoder = my_model()
autoencoder.load_state_dict(torch.load("model/autoencoder_weights.pth", map_location=device))
autoencoder.to(device)
autoencoder.eval()
out,latent,_ = autoencoder(faces)
reconstructed = out.detach().cpu()
show_faces(faces.cpu(), reconstructed)