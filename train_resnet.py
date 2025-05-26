import torch
from torch.utils.data import DataLoader, random_split
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
from torch.optim.lr_scheduler import StepLR,LambdaLR


def dynamic_step_fn(e):
    if e == 0:
        return 1
    return 0.1 / e


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
LEARNING_RATE = 0.01
NEW_MODEL = True
STEP_SCHEDULER = 1
GAMMA = 0.1
# ==========================


if NEW_MODEL:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(selected_attrs))
    model = model.to(device)
else:
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(selected_attrs))
    model.load_state_dict(torch.load("model/resnet50.pth"))
    model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = CelebADataset(attr_path, image_dir, selected_attrs, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = LambdaLR(optimizer, lr_lambda=dynamic_step_fn)

for epoch in range(EPOHS):
    model.train()
    running_loss = 0.0
    loop = tqdm.tqdm(train_loader)
    count_step = 0
    min_loss = 1000
    schelk = 0

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            count_step = 0
        else:
            count_step += 1
            if count_step >= 200:
                schelk += 1
                scheduler.step()
                count_step = 0

        running_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        loop.set_description(f"loss: {loss.item():.4f} lr: {current_lr:.6f} min_loss: {min_loss:.4f}")

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{EPOHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")



torch.save(model.state_dict(), "model/resnet18.pth")
model.eval()
all_preds = []
all_labels = []


# Метрика по каждому признаку
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

for i, attr in enumerate(selected_attrs):
    acc = accuracy_score(y_true[:, i], y_pred[:, i])
    print(f"{attr}: {acc:.4f}")
