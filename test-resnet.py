import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from Data_set import CelebADataset

image_dir = r'C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba'
attr_path = r"C:\Users\admin\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\list_attr_celeba.csv"

selected_attrs = [
    'Attractive', 'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    'Chubby', 'Gray_Hair', 'Mustache', 'Receding_Hairline', 'Sideburns',
    'Straight_Hair', 'Wavy_Hair', 'Young'
]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

DS = CelebADataset(attr_path, image_dir, selected_attrs, transform)


attrs_ru = ['Привлекательный', 'Лысый', 'Чёрные волосы', 'Светлые волосы', 'Каштановые волосы',
            'Пухлый', 'Седой', 'Усы', 'Залысины',
            'Бакенбарды', 'Прямые волосы', 'Волнистые волосы', 'Молодой']

# Загружаем модель
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, len(selected_attrs))
model.load_state_dict(torch.load("model/resnet50.pth", map_location=torch.device('cpu')))
model.eval()

img, label = DS[120000]
img_tensor = img.unsqueeze(0)

# Предсказание
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.sigmoid(outputs)[0]
    preds = (probs > 0.4).int()

# Получаем список предсказанных признаков
predicted = [ru for ru, p in zip(attrs_ru, preds) if p == 1]
fact = [ru for ru, p in zip(attrs_ru, label) if p == 1]

img = img.squeeze(0).permute(1, 2, 0).numpy()
print(f"предсказаные признаки:{predicted}")
print(f"вектор {probs}")
print(f"факт {fact}")
# Показываем результат
plt.imshow(img)
plt.axis("off")
plt.title("Предсказанные признаки")
plt.show()
