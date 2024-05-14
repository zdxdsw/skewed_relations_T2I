from PIL import Image
import torch

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def cycle(dl):
    while True:
        for data in dl:
            yield data

COLORS = {
    "red": "#ff0000", 
    "blue": "#0000ff", 
    "green": "#008000",
    "black": "#000000",
    "pink": "#ff69b4", 
    "yellow": "#ffff00",
    "purple": "#800080",
    "orange": "#FFA500",
    "cyan": "#00FFFF",
    #"brown": "#A52A2A",
    "khaki": "#BDB76B",
    "grey": "#808080",
    "lime":"#00FF00",
}

def hex_to_rgb(hex):
    h = hex.lstrip('#')
    if h == "000000": return (0.01, 0.01, 0.01)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

import numpy as np
from numpy.linalg import norm
def cos_sim(A, B):
    return np.sum(A*B, axis=1)/(norm(A, axis=1)*norm(B, axis=1))
def mse(A, B):
    return np.sum(np.square(A - B), axis=1) / 3

def L1(A, B):
    return np.sum(np.abs(A - B), axis=1) / 3


""" model summary helper """
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "#Params", "Param shape"])
    total_params = 0
    for name, parameter in model.named_parameters():
        #if not parameter.requires_grad: continue
        params = parameter.numel()
        param_shape = list(parameter.shape)
        table.add_row([name, params, param_shape])
        total_params+=params
    print(table)
    print(f"Total Params: {total_params}")