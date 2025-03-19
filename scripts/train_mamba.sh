set -ex
python -m visdom.server
python train.py --dataroot [dataset root] --name [experiment_name] --phase train --which_epoch latest
