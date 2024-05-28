# Prepare environment

pytorch version: 1.8.1+cu111
```
apt -y install python3-tk
```

```
pip install timm==0.3.2
pip install einops
```

# dataset

> https://drive.google.com/file/d/1qULO6dt0rjHTZcXP62FT4cqPq60XmkDe/view?usp=sharing


# trian

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

## pretrain

use all the test data and train data to pretrain the model
```
cd UM-MAE
make pretrain
```

## 2 classes

Categorize the dataset into two categories, 0,1, with 0 representing normal data and 1 representing abnormal data.
then run the following code

```
make finetunec2
```

```
make testc2
```


## 8 classes

Categorize the dataset into 0-7, 8 classes, each representing one type of data, if the image has multiple defects, it will be placed into two classes of data

```
make finetunec8
```

```
make testc8
```


## 9 classes

Categorize the dataset into 0-8, 9 categories, data with 2 defects will be placed in the ninth category

```
make finetune9
```

```
make testc9
```
