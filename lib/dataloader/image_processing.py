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

