import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir, attrs, transform=None):
        df = pd.read_csv(csv_file)
        df = df[df['Blurry'] != 1]

        self.data = df
        self.img_dir = img_dir
        self.attrs = attrs
        self.transform = transform

        self.data[self.attrs] = (self.data[self.attrs] + 1) // 2
        # Явно: привести к числовому типу
        self.data[self.attrs] = self.data[self.attrs].apply(pd.to_numeric, errors='coerce')
        # Убрать строки с пропущенными значениями, если они остались
        self.data = self.data.dropna(subset=self.attrs)
        self.data[self.attrs] = self.data[self.attrs].astype(int)
        # self.data = self.data.iloc[:len(self.data) // 2].reset_index(drop=True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels_np = np.array(self.data.iloc[idx][self.attrs].values, dtype=np.float32)
        labels = torch.from_numpy(labels_np)
        return image, labels
