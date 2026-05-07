"""
preprocess.py
-------------
Image preprocessing pipeline - Streamlit Cloud compatible.
"""

import numpy as np
from PIL import Image
import logging

IMG_SIZE = (224, 224)
MEAN     = np.array([0.485, 0.456, 0.406])
STD      = np.array([0.229, 0.224, 0.225])
logger   = logging.getLogger(__name__)


def load_image(image_input) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    return Image.open(image_input).convert("RGB")


def resize_image(img, size=IMG_SIZE):
    return img.resize(size, Image.LANCZOS)


def normalize_array(arr):
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.astype(np.float32)


def preprocess_image(image_input) -> np.ndarray:
    img = load_image(image_input)
    img = resize_image(img)
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = normalize_array(arr)
    return np.expand_dims(arr, axis=0)


def pil_to_display_array(image_input) -> np.ndarray:
    img = load_image(image_input)
    img = resize_image(img)
    return np.array(img)


def get_train_augmentation_params():
    return dict(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
        rescale=1.0 / 255.0,
    )


def get_val_augmentation_params():
    return {"rescale": 1.0 / 255.0}
