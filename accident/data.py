import csv
import os

import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def cnn_transform(image_size):
    """Preprocessing used to train the CNN: resize, scale to [0, 1], normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


def denormalize(image_tensor):
    """Inverse of :func:`cnn_transform` normalization, returns an ``[H, W, 3]`` array in [0, 1]."""
    return (image_tensor.cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def load_image_folder(data_dir, image_size=(512, 512)):
    """Loads ``data_dir/<label>/*.jpg|png`` where ``<label>`` is an integer class id.

    Returns ``(images [N, 3, H, W], labels [N])`` in a deterministic (sorted) order.
    """
    transform = cnn_transform(image_size)
    images, labels = [], []
    for label_dir in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label_dir)
        if not os.path.isdir(label_path):
            continue
        for name in sorted(os.listdir(label_path)):
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(label_path, name)
            try:
                images.append(transform(Image.open(path).convert("RGB")))
                labels.append(int(label_dir))
            except OSError as e:
                print(f"Skipping unreadable image {path}: {e}")
    return torch.stack(images), torch.tensor(labels)


def list_clips(root):
    """Sorted clip folder names under ``root`` (one folder of frames per clip)."""
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def last_frames(clip_dir, num_frames):
    """Paths of the last ``num_frames`` frames of a clip (frames sorted by filename)."""
    frames = sorted(f for f in os.listdir(clip_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    if len(frames) < num_frames:
        raise ValueError(f"{clip_dir} has {len(frames)} frames, {num_frames} required")
    return [os.path.join(clip_dir, f) for f in frames[-num_frames:]]


def read_labels(csv_path, key="file_name", value="risk"):
    """Reads a ``file_name,risk`` CSV into ``{clip_name: float(risk)}``."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row[key]: float(row[value]) for row in csv.DictReader(f)}
