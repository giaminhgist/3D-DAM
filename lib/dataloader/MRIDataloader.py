import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchio as tio
from .image_processing import normalise_zero_one, YH_reshape_image, reshape_zero_padding
from patchify import patchify, unpatchify


# geometric_transforms = {
#     tio.RandomFlip(axes=['Left-Right'], flip_probability=1): 0.25,
#     tio.RandomFlip(axes=['Anterior-Posterior'], flip_probability=1): 0.25,
#     tio.RandomFlip(axes=['Inferior-Superior'], flip_probability=1): 0.5,
# }
#
# transform = tio.Compose([
#     tio.OneOf(geometric_transforms, p=0.5),
#     tio.RandomBiasField(coefficients=0.2, p=0.5),
#     tio.RandomSwap(patch_size=10, num_iterations=100, p=0.5)
# ])


class MRIDataset(Dataset):
    def __init__(self, image_paths, label_dict, feature_dict,
                 # transformer=transform,
                 is_patch=True,
                 patch_size=32,
                 task='AD_CN_MCI',
                 ):
        # self.transformer = transformer
        self.is_patch = is_patch
        self.patch_size = patch_size

        self.image_paths = image_paths
        self.label_dict = label_dict
        self.feature_dict = feature_dict
        if task == 'AD_CN_MCI':
            classes = ['AD', 'CN', 'MCI']
        elif task == 'pMCI_sMCI':
            classes = ['pMCI', 'sMCI']
        elif task == 'CN_MCI':
            classes = ['CN', 'MCI']
        elif task == 'AD_MCI':
            classes = ['AD', 'MCI']
        else:
            classes = ['AD', 'CN']

        self.idx_to_class = {i: j for i, j in enumerate(classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image_id = image_filepath.split('/')[-1]
        label = self.label_dict[image_id]
        label = self.class_to_idx[label]
        feature = self.feature_dict[image_id]

        # subject_id = image_filepath.split('/')[-4] + image_filepath.split('/')[-3]
        image_original = nib.load(image_filepath).get_fdata()
        image = normalise_zero_one(image_original)

        # image = reshape_zero_padding(image)
        # image = np.expand_dims(image, axis=0)

        if feature is not None:
            # if self.is_patch:
            #     image = patchify(image, (self.patch_size, self.patch_size, self.patch_size), step=self.patch_size)
            #     image = image.reshape(-1, self.patch_size, self.patch_size, self.patch_size)
            #     return image, feature, label
            # else:
            #     return image, feature, label
            return image, feature, label
        else:
            return image, label


class DataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)

        return data

    def __len__(self):
        return len(self.data_loader)
