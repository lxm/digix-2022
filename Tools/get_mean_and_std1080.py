# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 22:20:17 2022

@author: LocalAdmin
"""

from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms

def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0, #error for images size not equal
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/train_image_crop1080', transform=transforms.ToTensor())
    print(get_mean_and_std(train_dataset))