import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import ChangeDataset
from models.model import ChangeModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ChangeDataset("data/test")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = ChangeModel().to(device)

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

model.eval()


def compute_iou(pred, target):
    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    return intersection / (union + 1e-6)


ious = []

for i, (x, y) in enumerate(loader):
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        pred = model(x)

    pred = (pred > 0.5).float()

    pred_np = pred.squeeze().cpu().numpy()
    gt_np = y.squeeze().cpu().numpy()

    iou = compute_iou(pred_np, gt_np)

    ious.append(iou)

    # visualize some samples
    if i < 5:
        input_img = x[0][:3].permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(input_img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(gt_np, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred_np, cmap='gray')
        plt.axis("off")

        plt.show()

print(f"Average IoU: {np.mean(ious)}")