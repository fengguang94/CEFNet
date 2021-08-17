import torch
import time
import torch.nn as nn
import random
import os
import numpy as np

def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
seed_torch()

d = 1
m = nn.Conv2d(256, 256, 3, 1, padding=d, dilation=d, bias=False)
m.cuda()
total = 0.0

for _ in range(100):
    i = torch.rand(1, 256, 80, 45).cuda()
    s = time.time()
    r = m(i)
    torch.cuda.synchronize()
    e = time.time()
    total += (e-s)
print(torch.version.__version__)
print(total/100)