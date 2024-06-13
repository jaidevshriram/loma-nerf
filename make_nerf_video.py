import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to your test images and transforms file
images_dir = "./data/lego/test"
transforms_path = "./data/lego/transforms_test.json"

# Load camera parameters
with open(transforms_path, "r") as f:
    transforms = json.load(f)

camera_angle_x = transforms["camera_angle_x"]
frames = transforms["frames"]


# Function to render images based on camera parameters
def render_images(images_dir, frames):
    rendered_images = []
    for frame in frames:
        file_path = os.path.join(
            images_dir, os.path.basename(frame["file_path"]) + ".png"
        )
        image = Image.open(file_path)
        rendered_images.append(np.array(image))
    return rendered_images


rendered_images = render_images(images_dir, frames)

# Path to save the video
video_path = "./logs/videos/nerf_rendering.mp4"

# Create a video from the rendered images
imageio.mimsave(video_path, rendered_images, fps=30)

# Optionally, display a few images
for i in range(0, len(rendered_images), 10):
    plt.imshow(rendered_images[i])
    plt.axis("off")
    plt.show()
