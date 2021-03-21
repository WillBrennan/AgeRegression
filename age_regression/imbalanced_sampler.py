from collections import defaultdict

import torch
import torch.utils.data as data

from .dataset import AllAgeFacesDataset


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset: AllAgeFacesDataset):
        self.num_samples = len(dataset)

        # distribution of classes in the dataset
        label_count = defaultdict(int)

        for label in dataset.labels:
            label_count[label['age']] += 1

        self.weights = [label_count[label['age']] for label in dataset.labels]
        self.weights = 1.0 / torch.DoubleTensor(self.weights)

    def __iter__(self):
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        yield from indices.tolist()

    def __len__(self):
        return self.num_samples
