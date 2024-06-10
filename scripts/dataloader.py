import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class NeRFDataset(Dataset):
    def __init__(self, root_dir, img_size=16, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for phase in ["train", "val", "test"]:
            phase_dir = os.path.join(self.root_dir, phase)
            images = sorted(
                [
                    f
                    for f in os.listdir(phase_dir)
                    if f.endswith(".png") and "depth" not in f
                ]
            )
            depth_maps = sorted(
                [
                    f
                    for f in os.listdir(phase_dir)
                    if f.endswith(".png") and "depth" in f
                ]
            )

            for img, depth in zip(images, depth_maps):
                img_path = os.path.join(phase_dir, img)
                depth_path = os.path.join(phase_dir, depth)
                data.append((img_path, depth_path, phase))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, depth_path, phase = self.data[idx]

        # Debugging: Print file paths
        print(f"Loading image from: {img_path}")
        print(f"Loading depth map from: {depth_path}")

        image = Image.open(img_path).resize((self.img_size, self.img_size))
        depth = Image.open(depth_path).resize((self.img_size, self.img_size))

        # Debugging: Show image and depth info
        print(f"Image size after loading: {image.size}")
        print(f"Depth size after loading: {depth.size}")

        image = np.array(image, dtype=np.float32) / 255.0
        depth = np.array(depth, dtype=np.float32) / 255.0

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        sample = {"image": image, "depth": depth, "phase": phase}
        return sample
