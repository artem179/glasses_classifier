import torch
import numpy as np
from torchvision import transforms

mean, std = ([0.4437454317480777, 0.47589344395934596, 0.5517830162603856],
             [0.24644172796403027, 0.24670957395057505, 0.22897207651616666]) # Calculated mean and std on CelebA cropped dataset

data_transforms = {
    'train' : transforms.Compose([
              transforms.ToPILImage(),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean=mean,
                                     std=std),]),
    'val' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                     std=std),])}

class ToTensorUint(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeUint(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor