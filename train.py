import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.utils import data
from ignite import engine
from ignite import metrics
from torch.utils import tensorboard

from age_regression import AllAgeFacesDataset
from age_regression import AgeRegressionModel

from age_regression import create_data_loaders
from age_regression import attach_lr_scheduler
from age_regression import attach_training_logger
from age_regression import attach_model_checkpoint
from age_regression import attach_metric_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--model-tag', type=str, required=True)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--initial-lr', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=18)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'running training on {device}')

    logging.info('creating dataset and data loaders')
    train_dataset = AllAgeFacesDataset(args.train, use_augmentation=True)
    val_dataset = AllAgeFacesDataset(args.val, use_augmentation=False)

    train_loader, train_metrics_loader, val_metrics_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    logging.info(f'creating model and optimizer with initial lr of {args.initial_lr}')
    model = AgeRegressionModel().to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.initial_lr)

    logging.info('creating trainer and evaluator engines')

    trainer = engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        device=device,
        non_blocking=True,
    )

    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={
            'loss': metrics.Loss(nn.MSELoss()),
            'mse': metrics.MeanSquaredError(),
            'mae': metrics.MeanAbsoluteError(),
        },
        device=device,
        non_blocking=True,
    )

    logging.info(f'creating summary writer with tag {args.model_tag}')
    writer = tensorboard.SummaryWriter(log_dir=f'logs/{args.model_tag}')

    logging.info('attaching lr scheduler')
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    attach_lr_scheduler(trainer, lr_scheduler, writer)

    logging.info('attaching event driven calls')
    attach_model_checkpoint(trainer, {args.model_tag: model})
    attach_training_logger(trainer, writer=writer)

    attach_metric_logger(trainer, evaluator, 'train', train_metrics_loader, writer)
    attach_metric_logger(trainer, evaluator, 'val', val_metrics_loader, writer)

    logging.info('training...')
    trainer.run(train_loader, max_epochs=args.num_epochs)
