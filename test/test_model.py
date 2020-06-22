from age_regression import AgeRegressionModel
import torch


def test_AgeRegressionModel():
    model = AgeRegressionModel()

    images = torch.zeros(20, 3, 224, 224)
    x_age = model(images)

    assert x_age.shape == torch.Size([20, 1])
    assert x_age.dtype == torch.float32
