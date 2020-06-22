import argparse
import logging
import pathlib
import functools

import cv2
import torch

from age_regression import AgeRegressionModel
from age_regression import create_image_transform
from age_regression import denormalize_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logging.info(f'loading AgeRegressionModel from {args.model}')
    model = AgeRegressionModel()
    model.load_state_dict(torch.load(args.model))
    model.cuda().eval()

    logging.info(f'evaluating images from {args.images}')
    image_dir = pathlib.Path(args.images)

    image_transform = create_image_transform(use_augmentation=False)

    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        image = image_transform(image_file)

        with torch.no_grad():
            image = image.cuda().unsqueeze(0)
            pred_age = model(image)

            image = image[0]
            pred_age = pred_age[0].item()

        logging.info(f'image: {image_file} age: {pred_age:.1f}')
        cv2.imshow('image', denormalize_image(image))

        if cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            exit()
