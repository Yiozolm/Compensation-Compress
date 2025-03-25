import torch
import torchvision.transforms as T

from PIL import Image
import numpy as np


class PILToTensor(object):
    def __call__(self, pic):
        assert isinstance(pic, Image.Image), "PILToTensor: Please input a PIL image."

        img = torch.as_tensor(np.array(pic))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute((2, 0, 1)).float()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_transforms(transform_type):
    if transform_type == "p256":
        transform = T.Compose([
            T.RandomCrop(256),
            PILToTensor()
        ])
    elif transform_type == "p256_centercrop":
        transform = T.Compose([
            T.CenterCrop(256),
            PILToTensor()
        ])

    else:
        raise Exception("No existing transform type {}.".format(transform_type))

    return transform
