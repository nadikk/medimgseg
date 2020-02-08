from torch.utils.data.dataset import Dataset
from typing import Optional, Callable
from glob import glob
import nibabel as nib
import numpy as np
import os

def glob_imgs(dir:str):
    """
    grab all (not hidden) nii files in a directory and sort them
    """
    return sorted([f for f in glob(os.path.join(dir, "*.nii*")) if not f.startswith('.')])

class NiftiDataset(Dataset):
    """
    dataset class for reading Nifti files

    Args:
        images_dir (str): path to train images
        labels_dir (str): path to train labels
        transform (Callable): transform to apply to both source and target images
    """

    def __init__(self, images_dir:str, labels_dir:str, transform:Optional[Callable]=None):
        self.images_dir, self.labels_dir = images_dir, labels_dir
        self.images, self.labels = glob_imgs(images_dir), glob_imgs(labels_dir)
        self.transform = transform
        if len(self.images) != len(self.labels) or len(self.images) == 0:
            raise ValueError("Number of images and labels must be equal and non-zero")

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx:int):
        imgs, labs = self.images[idx], self.labels[idx]
        sample = (nib.load(imgs).get_fdata(dtype=np.float32), nib.load(labs).get_fdata(dtype=np.float32))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample