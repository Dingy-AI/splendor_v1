import torch
import numpy as np
from splendor_v1.env.env import SplendorEnv
print(torch.__version__)
print(np.__version__)


splendor = SplendorEnv()


clone = splendor.clone()


clone.seed += 100

print(splendor.seed)
print(clone.seed)
