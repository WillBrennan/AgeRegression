import argparse
import logging

import cv2
import numpy
import torch

from age_regression import AllAgeFacesDataset
from age_regression import denormalize_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--use-augmentation', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def log_stats(name, data):
    data_type = data.dtype
    data = data.float()
    logging.info(f'{name} - {data.shape} - {data_type} - min: {data.min()} mean: {data.mean()} max: {data.max()}')


if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    dataset = AllAgeFacesDataset(args.dataset, args.use_augmentation)

    num_samples = len(dataset)
    for idx in range(num_samples):
        logging.info(f'showing {(idx + 1)} of {num_samples} samples')
        image, age = dataset[idx]

        log_stats('image', image)
        logging.info(f'age: {age}')

        image = denormalize_image(image)
        cv2.imshow('image', image)

        if cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            exit()
