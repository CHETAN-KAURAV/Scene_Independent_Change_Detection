import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ChangeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for scene in os.listdir(root_dir):
            input_dir = os.path.join(root_dir, scene, "input")
            gt_dir = os.path.join(root_dir, scene, "groundtruth")

            frames = sorted(os.listdir(input_dir))

            for i in range(2, len(frames)):
                f1 = frames[i - 2]
                f2 = frames[i - 1]
                f3 = frames[i]

                gt_name = f3.replace("in", "gt").replace(".jpg", ".png")
                gt_path = os.path.join(gt_dir, gt_name)

                if not os.path.exists(gt_path):
                    continue

                self.samples.append(
                    (input_dir, gt_dir, f1, f2, f3, gt_name)
                )

        print(f"Loaded {len(self.samples)} filtered samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_dir, gt_dir, f1, f2, f3, gt_name = self.samples[idx]

        img_t2 = cv2.imread(os.path.join(input_dir, f1))
        img_t1 = cv2.imread(os.path.join(input_dir, f2))
        img_t = cv2.imread(os.path.join(input_dir, f3))

        gt = cv2.imread(os.path.join(gt_dir, gt_name), 0)

        # resize
        img_t2 = cv2.resize(img_t2, (224, 224))
        img_t1 = cv2.resize(img_t1, (224, 224))
        img_t = cv2.resize(img_t, (224, 224))
        gt = cv2.resize(gt, (224, 224))

        # temporal differencing (BEST VERSION)
        diff1 = cv2.absdiff(img_t, img_t1)
        diff2 = cv2.absdiff(img_t, img_t2)

        x = np.concatenate([diff1, diff2], axis=2)

        gt = (gt == 255).astype(np.float32)

        x = torch.tensor(x).permute(2, 0, 1).float() / 255.0
        gt = torch.tensor(gt).unsqueeze(0)

        return x, gt