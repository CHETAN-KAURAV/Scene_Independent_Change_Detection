import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np


class PretrainDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for scene in os.listdir(root_dir):
            input_dir = os.path.join(root_dir, scene, "input")
            frames = sorted(os.listdir(input_dir))

            for i in range(len(frames) - 1):
                self.samples.append((input_dir, frames[i], frames[i+1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, f1, f2 = self.samples[idx]

        img1 = cv2.imread(os.path.join(folder, f1))
        img2 = cv2.imread(os.path.join(folder, f2))

        img1 = cv2.resize(img1, (224,224))
        img2 = cv2.resize(img2, (224,224))

        # random order
        if np.random.rand() > 0.5:
            x = np.concatenate([img1, img2], axis=2)
            label = 1
        else:
            x = np.concatenate([img2, img1], axis=2)
            label = 0

        x = torch.tensor(x).permute(2,0,1).float() / 255.0
        return x, torch.tensor(label)


class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        self.encoder.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.encoder(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PretrainDataset("../data/train")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = PretrainModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Pretrain Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.encoder.state_dict(), "../pretrained_encoder.pth")