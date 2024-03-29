import logging
import functools

from torch.utils import data

from .imbalanced_sampler import ImbalancedDatasetSampler


def create_data_loaders(train_dataset: data.Dataset, val_dataset: data.Dataset, num_workers: int, batch_size: int):
    logging.info(f'creating dataloaders with {num_workers} workers and a batch-size of {batch_size}')

    fn_dataloader = functools.partial(
        data.DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    imbalanced_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = fn_dataloader(train_dataset, sampler=imbalanced_sampler)

    train_metrics_sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=len(val_dataset))
    train_metrics_loader = fn_dataloader(train_dataset, sampler=train_metrics_sampler)

    val_metrics_loader = fn_dataloader(val_dataset)

    return train_loader, train_metrics_loader, val_metrics_loader
