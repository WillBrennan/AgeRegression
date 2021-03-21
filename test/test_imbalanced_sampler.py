from age_regression.imbalanced_sampler import ImbalancedDatasetSampler

from collections import defaultdict
import torch


class Dataset:
    def __init__(self, labels):
        self.labels = [{'age': i} for i in labels]

    def __len__(self):
        return len(self.labels)


def test_ImbalancedDatasetSampler():
    torch.manual_seed(0)

    num_values = 10000
    labels = []

    for idx in range(1, 7):
        num_count = idx * num_values
        labels += num_count * [idx]

    dataset = Dataset(labels)

    sampler = ImbalancedDatasetSampler(dataset)
    gen_labels = list(dataset.labels[idx]['age'] for idx in iter(sampler))

    label_counts = defaultdict(int)
    for label in gen_labels:
        label_counts[label] += 1

    expected = sum(range(1, 7)) * num_values / 6
    print(expected)
    print(label_counts)

    chi2 = sum(pow(label_count - expected, 2) / expected for label_count in label_counts.values())

    # p=0.99
    critical_value = 1.239
    assert chi2 < critical_value, 'not a uniform distribution with p=0.99'
