# this is the main entrypoint
# as we describe in the paper, we compute the flops over the first 100 images
# on COCO val2017, and report the average result
# great thanks for fmassa!

# Run this python file with a config file, such as:
# python tools/compute_flops.py experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc5_mh8.yaml

import torch
import time
import torchvision

import numpy as np
import tqdm

from flop_count import flop_count

import yaml
from yacs.config import CfgNode

def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()

def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()

import _init_paths
import dataset
import models
import sys


file_name = sys.argv[1]
f = open(file_name, 'r')

cfg = CfgNode(yaml.load(f))
cfg.DATASET.DATA_FORMAT = 'jpg'
cfg.DATASET.NUM_JOINTS_HALF_BODY = 8
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
cfg.TEST.SOFT_NMS = False
cfg.MODEL.INIT_WEIGHTS = True

import torchvision.transforms as transforms
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

import numpy as np
import random
seed = 22
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

  
dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

images = []
for idx in range(100):
    img, _, _, _ = dataset[idx]
    images.append(img)

device = torch.device('cuda')
results = {}
_name = file_name.split('.')[-2]

for model_name in [_name]:
    
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        ckpt_state_dict = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(ckpt_state_dict, strict=True)

    a = torch.randn(1, 3, 256, 192)
    model.to(device)
    print("model params:{:.3f}M (/1000^2)".format(sum([p.numel() for p in model.parameters()])/1000**2))
    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(images):
            inputs = torch.cat([img.to(device)]).unsqueeze(0)
            res = flop_count(model, (inputs,))
            t = measure_time(model, inputs)
            tmp.append(sum(res.values()))
            tmp2.append(t)

    results[model_name] = {'flops': fmt_res(np.array(tmp)), 'time': fmt_res(np.array(tmp2))}


print('=============================')
print('')
for r in results:
    print(r)
    for k, v in results[r].items():
        print(' ', k, ':', v)

print("FPS=", 1/results[model_name]['time'][0])