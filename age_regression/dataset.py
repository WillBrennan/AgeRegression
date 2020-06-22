import logging
import pathlib

from PIL import Image
import numpy
import torch
from torch.utils import data
import torchvision.transforms as transforms


def _load_image(image_path: str):
    with open(image_path, 'rb') as f:
        return Image.open(f).convert('RGB')


def create_image_transform(use_augmentation: bool):
    image_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]

    if use_augmentation:
        aug_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]

        image_transforms = [transforms.RandomOrder(aug_transforms)] + image_transforms
    else:
        image_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ] + image_transforms

    image_transforms = [_load_image] + image_transforms

    return transforms.Compose(image_transforms)


def denormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    assert isinstance(image, numpy.ndarray)
    image = numpy.transpose(image, (1, 2, 0))

    image_mean = numpy.array((0.485, 0.456, 0.406))
    image_std = numpy.array((0.229, 0.224, 0.225))
    image = image * image_std + image_mean

    image = (255 * image).astype(numpy.uint8)
    image = image[..., ::-1]
    return image


class AllAgeFacesDataset(data.Dataset):
    def __init__(self, image_dir: str, use_augmentation: bool):
        image_dir = pathlib.Path(image_dir)
        self.image_paths = list(image_dir.rglob('*.jpg'))
        assert len(self.image_paths) == 13322

        self.max_age = 0

        self.labels = []
        for image_path in self.image_paths:
            image_name = image_path.with_suffix('').name
            gender, age = [i for i in image_name.split('A')]
            is_female = int(gender) <= 7380
            age = int(age)

            self.max_age = max(self.max_age, age)

            self.labels += [{'age': age, 'is_female': is_female}]

        self.image_transform = create_image_transform(use_augmentation)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = self.image_transform(image_path)
        age = torch.tensor([label['age']], dtype=torch.float32)

        return image, age
