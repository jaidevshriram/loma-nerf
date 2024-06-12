import os
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pdb


class NeRFDataset(Dataset):
    def __init__(self, root_dir, img_size=16, phase="train"):
        self.root_dir = root_dir
        self.img_size = img_size
        self.phase = phase
        self.data = self._load_data()

    def _load_data(self):
        data = []

        phase_dir = os.path.join(self.root_dir, self.phase)
        transform_path = os.path.join(self.root_dir, f"transforms_{self.phase}.json")

        with open(transform_path, "r") as f:
            transforms = json.load(f)

        self.camera_angle_x = float(transforms["camera_angle_x"])

        frames = transforms["frames"]

        for frame in frames:
            img_path = os.path.join(self.root_dir, frame["file_path"] + ".png")
            transform_matrix = np.array(frame["transform_matrix"])

            data.append((img_path, transform_matrix))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, transform_matrix = self.data[idx]

        # Debugging: Print file paths
        # print(f"Loading image from: {img_path}")

        image = Image.open(img_path).resize((self.img_size, self.img_size)).convert("RGB")

        # Debugging: Show image and depth info
        # print(f"Image size after loading: {image.size}")

        image = np.asarray(image, dtype=np.float32) / 255.0

        # print(f"Image shape after loading: {image.shape}")

        sample = {"image": image, "pose": transform_matrix, "focal_length": 0.5 / np.tan(0.5 * self.camera_angle_x)}
        return sample

if __name__ == "__main__":
    dataset = NeRFDataset("data/lego")