import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from Data_set import CelebADataset
from dotenv import load_dotenv
import os

load_dotenv()

image_dir = os.getenv("image_dir")
attr_path = os.getenv("attr_path")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selected_attrs = [
    'Attractive', 'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    'Chubby', 'Gray_Hair', 'Mustache', 'Receding_Hairline', 'Sideburns',
    'Straight_Hair', 'Wavy_Hair', 'Young'
]

# ==========================
EPOHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NEW_MODEL = False
# ==========================
if NEW_MODEL:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(selected_attrs))
    model = model.to(device)
else:
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(selected_attrs))
    model.load_state_dict(torch.load("resnet18.pth"))
    model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = CelebADataset(attr_path, image_dir, selected_attrs, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOHS):
    model.train()
    running_loss = 0.0
    loop = tqdm.tqdm(dataloader)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{EPOHS}] - Loss: {running_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "resnet18.pth")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    loop = tqdm.tqdm(dataloader)
    for images, labels in loop:
        images = images.to(device)
        outputs = torch.sigmoid(model(images))
        preds = (outputs > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_labels)

# Метрика по каждому признаку
for i, attr in enumerate(selected_attrs):
    acc = accuracy_score(y_true[:, i], y_pred[:, i])
    print(f"{attr}: {acc:.4f}")
