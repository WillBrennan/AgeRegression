# Age Regression
## Overview
I was at a supermarket buying wine and I used a self-checkout, suprisingly it didn't ask an assistant to verify my age! Then I noticed it had a camera in the bezzle. So then I wondered... could we estimate the age of a person from a picture of their face and to what degree of confidence.

In this project we attempt to estimate the age of a person from a picture of their face. We do this by training on [All-Ages-Faces-Dataset](https://github.com/JingchunCheng/All-Age-Faces-Dataset) and regress the age with a MSE loss. This dataset has a heavy racial bias, and impacts the models performance.

![Age Regression](https://raw.githubusercontent.com/WillBrennan/AgeRegression/master/pretrained/age_regression.png)

## Getting Started
This project uses conda to manage its enviroment; once conda is installed we create the enviroment and activate it, 
```bash
conda env create -f enviroment.yml
conda activate age_regression
```
. On windows; powershell needs to be initialised and the execution policy needs to be modified. 
```bash
conda init powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

. This repo uses git-lfs to store the models, make sure the git-lfs files have been pulled using, 

```bash
git lfs pull
```

To run age prediction on a directory of images run, 

```bash
python evaluate_images.py --model pretrained/model_age_regression_resnext101_20.pth --images ~/code/datasets/faces/val
```

## Performance
When we do a joint-plot between predicted and ground-truth age we can see the model has a strong correlation between the two but there are the occasional outliers. 

![joint_plot](https://raw.githubusercontent.com/WillBrennan/AgeRegression/master/pretrained/joint_plot.png)

We can see the dataset has a large age inbalance, mostly featuring samples from people 25~35. As we're trying to maximise our performance on this dataset its has been ignored. In the future, this dataset should be resampled before training to give a uniform age distribution. 

![gt_ages_dist](https://raw.githubusercontent.com/WillBrennan/AgeRegression/master/pretrained/gt_ages_dist.png)
![pred_ages_dist](https://raw.githubusercontent.com/WillBrennan/AgeRegression/master/pretrained/pred_ages_dist.png)


When we plot the MAE at each age group. We can see our age group with the smallest MAE is 30-35 which corresponds to our predominant age-group in the dataset. Where we have very little data we can see larger prediction errors. Its possibly interesting that the groups for ages <15 also have lower MAE, possibly because they're easier cases.

![mae_for_age](https://raw.githubusercontent.com/WillBrennan/AgeRegression/master/pretrained/mae_for_age.png)


