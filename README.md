# OCSSNet
## Installation
``` shell
# pip install required packages
conda create -n mambayolo -y python=3.11
conda activate mambayolo
pip3 install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

## Training

```shell
python train.py --task train \
  --data ultralytics/cfg/datasets/RSDDs.yaml \
  --config ultralytics/cfg/models/ocss/OCSS-T.yaml \
  --amp --project ./output_dir/rsdd --name OCSS
```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan from [VMamba](https://github.com/MzeroMiko/VMamba).
