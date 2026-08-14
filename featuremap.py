import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
from collections import OrderedDict

# # # 加载模型
# # # model = torch.load('./save/model.pkl').to(torch.device('cpu'))
# model_dict = torch.load('/root/output_dir/mscoco/mambayolo/weights/best.pt',map_location=torch.device('cpu'))
# 如果有 GPU 可用，将模型加载到 GPU 上，否则加载到 CPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并将其移动到目标设备
model_dict = torch.load('/root/output_dir/mscoco/mambayolo/weights/best.pt', map_location=device)
model = model_dict['model']
# print(model)
# print(model.model[0])  # SimpleStem是第0层
print(model.model[1])














