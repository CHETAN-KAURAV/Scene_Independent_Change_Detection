import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import ChangeDataset
from models.model import ChangeModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ChangeDataset("data/train")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = ChangeModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()

    total_loss = 0

    loop = tqdm(loader)

    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = F.binary_cross_entropy(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader)}")

torch.save(model.state_dict(), "model.pth")

print("Model saved as model.pth")