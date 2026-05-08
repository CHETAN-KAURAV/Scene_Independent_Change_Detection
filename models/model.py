import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ChangeModel(nn.Module):
    def __init__(self):
        super(ChangeModel, self).__init__()

        resnet = models.resnet18(pretrained=True)

        # 6-channel input
        resnet.conv1 = nn.Conv2d(
            6,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # U-Net style decoder (BEST VERSION)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        x = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        x = torch.sigmoid(x)

        return x