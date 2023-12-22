import numpy as np
import torch.nn.functional as F


# Preprocessing function
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def reshape_zero_padding(img, target_shape=224):
    s_1 = int((target_shape - img.shape[0]) / 2)
    s_2 = target_shape - img.shape[0] - s_1
    c_1 = int((target_shape - img.shape[1]) / 2)
    c_2 = target_shape - img.shape[1] - c_1
    a_1 = int((target_shape - img.shape[2]) / 2)
    a_2 = target_shape - img.shape[2] - a_1
    img = np.pad(img, ((s_1, s_2), (c_1, c_2), (a_1, a_2)), 'constant', constant_values=0)
    return img


def YH_reshape_image(image, desired_size=None):
    if desired_size is None:
        desired_size = [224, 224, 224]
    padding = [(d - image.size(i)) // 2 for i, d in enumerate(desired_size)]
    padding = [p if p >= 0 else 0 for p in padding]  # if already greater, no padding needed
    padding_tuples = [(padding[2], padding[2] + (desired_size[2] - image.size(2)) % 2),
                      (padding[1], padding[1] + (desired_size[1] - image.size(1)) % 2),
                      (padding[0], padding[0] + (desired_size[0] - image.size(0)) % 2)]
    padding_flat = [p for sublist in padding_tuples for p in sublist]
    image_padded = F.pad(image, padding_flat, "constant", 0)
    return image_padded
