This is the code demo for the paper:  
_Dong, Y., Lu, X., Li, R., Song, W., van Arem, B. and Farah, H., 2023. Intelligent Anomaly Detection for Lane Rendering Using Transformer with Self-Supervised Pre-Training and Customized Fine-Tuning. [arXiv preprint arXiv:2312.04398](https://arxiv.org/abs/2312.04398)_.


# Prepare environment

pytorch version: 1.8.1+cu111
```
apt -y install python3-tk
```

```
pip install timm==0.3.2
pip install einops
```

# Dataset

> https://github.com/lxm/digix-2022/blob/master/digix-data-resize.tar.gz  


# Trian

file tree
```
├── UM-MAE
│   ├── DET
│   │   └── configs
│   ├── IN1K
│   ├── SEG
│   │   ├── configs
│   │   └── mmcv_custom
│   ├── __pycache__
│   ├── figs
│   ├── util
│   │   └── __pycache__
│   ├── visual
│   └── work_dirs
│       └── allimage
└── huaweidata
    ├── pretrain-allimage
    │   └── all
    ├── test_images
    └── train_image
```

## Pretrain

Use all the test data and train data(without labels) to pre-train the model
```
cd UM-MAE
make pretrain
```

## Treat it as a 2-class classification problem

Categorize the dataset into two categories, 0,1, with 0 representing normal image data and 1 representing abnormal instance.
then run the following code

```
make finetunec2
```

```
make testc2
```


## Treat it as an 8-class classification problem

Categorize the dataset into 0-7, in total 8 classes, each representing one type of data, if the image has multiple defects, it will be placed into multiple classes of data

```
make finetunec8
```

```
make testc8
```


## Treat it as a 9-class classification problem

Categorize the dataset into 0-8, 9 categories, data with each abnormal image instance (regardless of what type of anomaly is ) will be placed in the ninth category 

```
make finetune9
```

```
make testc9
```
