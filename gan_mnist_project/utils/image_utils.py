# utils/image_utils.py
import numpy as np
from PIL import Image


def save_images(images, path):
    images = (images + 1.0) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = images.reshape((len(images), 28, 28))

    grid_size = int(np.ceil(np.sqrt(len(images))))
    grid_image = Image.new('L', (28 * grid_size, 28 * grid_size))

    for i, image in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        image = Image.fromarray(image, mode='L')
        grid_image.paste(image, (col * 28, row * 28))

    grid_image.save(path)
