import os
import sys
import matplotlib.pyplot as plt

# Fix path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.dataset import ChangeDataset

# Correct absolute path
train_path = os.path.join(BASE_DIR, "data", "train")

dataset = ChangeDataset(train_path)

print("Dataset size:", len(dataset))

# show a meaningful sample
for i in range(len(dataset)):
    x, y = dataset[i]

    if y.sum() > 500:
        plt.imshow(y.squeeze(), cmap='gray')
        plt.title(f"Sample {i}")
        plt.show()
        break