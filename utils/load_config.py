import torch

if torch.cuda.is_available():
    cache_dir = "/data/songyao/models"
else:
    cache_dir = "/Users/songyao/.cache"
