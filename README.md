# DCM
An Efficient Infrared Image Colorization via Dual-Branch Feature Interactive Fusion

## Prerequisites
- python 3.7
- torch 1.13.1
- torchvision 0.14.1
- dominate
- visdom

## Dataset
We provide the [KAIST](https://github.com/SoonminHwang/rgbt-ped-detection) dataset and the [FLIR](https://www.flir.com/oem/adas/adas-dataset-form) dataset link.

## Trian
```
python train.py --dataroot [dataset root] --name [experiment_name] --phase train --which_epoch latest
```

## Test
```
python test.py --dataroot [dataset root] --name [experiment_name] --phase test --which_epoch latest
```

## Colorization results
### KAIST dataset
![KAIST](img/KAIST.png)


### FLIR dataset
![FLIR](img/FLIR.png)


## Acknowledgments
This code heavily borrowes from [FRAGAN](https://github.com/cyanymore/FRAGAN).
