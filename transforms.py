from typing import Tuple
from torch.nn import ConstantPad3d
import numpy as np
import torch

#if type(imgs) != torch.Tensor or type(labs) != torch.Tensor:
 #   raise ValueError("Images and labels must be transformed to tensors")

class ToTensor():
    """
    convert image and label in sample to tensors
    """

    def __call__(self, sample:Tuple[np.ndarray, np.ndarray]):
        imgs, labs = sample
        return (torch.from_numpy(imgs), torch.from_numpy(labs))

class AddChannel():
    """
    add empty first dimension to sample
    """

    def __call__(self, sample:Tuple[np.ndarray, np.ndarray]):
        imgs, labs = sample
        return (imgs.unsqueeze(0), labs.unsqueeze(0))

class Pad3d():
    """
    pad sample to be max size
    """
    def __init__(self, max_sizes:Tuple[int,int,int]):
        self.x_max, self.y_max, self.z_max = max_sizes[0], max_sizes[1], max_sizes[2]

    def __call__(self, sample:Tuple[np.ndarray, np.ndarray]):
        if len(sample[0].shape) == 4:
            x, y, z = sample[0].shape[1], sample[0].shape[2], sample[0].shape[3]
        else:
            x, y, z = sample[0].shape[0], sample[0].shape[1], sample[0].shape[2]
        diffs = [self.x_max-x, self.y_max-y, self.z_max-z]
        pad_sizes = []
        for diff in diffs:
            if (diff % 2):
                pad_sizes.append(int(diff/2)+1)
                pad_sizes.append(int(diff/2))
            else:
                pad_sizes.append(int(diff/2))
                pad_sizes.append(int(diff/2))
        left, right, top, bottom, front, back = pad_sizes[0], pad_sizes[1], pad_sizes[2], \
                                                pad_sizes[3], pad_sizes[4], pad_sizes[5]
        pad = ConstantPad3d((front, back, top, bottom, left, right), 0)
        return (pad(sample[0]), pad(sample[1]))
