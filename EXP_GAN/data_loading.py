import os
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join

import torchvision.transforms as transforms
import torchvision.datasets as dset


### Get Data ###
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class FolderWithImages(data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        super(FolderWithImages, self).__init__()
        self.image_filenames = [join(root, x)
                                for x in listdir(root) if is_image_file(x.lower())]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class ALICropAndScale(object):
    def __call__(self, img):
        return img.resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))



def get_data(image_size,dataset_name,data_root,train_flag=True):
    if dataset_name == 'cifar10':

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = dset.CIFAR10(root=data_root,
                               download=True,
                               train=train_flag,
                               transform=transform)

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)),
        ])

        dataset = dset.MNIST(root=data_root,
                             download=True,
                             train=train_flag,
                             transform=transform)


    elif dataset_name == 'celeba':
        imdir = 'CelebA/splits/train' if train_flag else 'CelebA/splits/val'
        dataroot = os.path.join(data_root, imdir)
        if image_size != 64:
            raise ValueError('the image size for CelebA dataset need to be 64!')

        dataset = FolderWithImages(root=dataroot,
                                   input_transform=transforms.Compose([
                                       ALICropAndScale(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]),
                                   target_transform=transforms.ToTensor()
                                   )

    return dataset
