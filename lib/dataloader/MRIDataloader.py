import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from .image_processing import normalise_zero_one, reshape_zero_padding


class MRIDataset(Dataset):
    def __init__(self, image_paths, label_dict, feature_dict,
                 task='AD_CN',
                 ):

        self.image_paths = image_paths
        self.label_dict = label_dict
        self.feature_dict = feature_dict

        if task == 'AD_CN':
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

        image_original = nib.load(image_filepath).get_fdata()
        image = normalise_zero_one(image_original)
        # image = reshape_zero_padding(image)
        # image = np.expand_dims(image, axis=0)
        
        return image, label
