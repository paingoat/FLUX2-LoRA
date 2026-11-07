"""Seed all training RNGs before entering AI Toolkit's regular CLI."""
import os
import random
import runpy
import sys

import numpy as np
import torch

seed = int(os.environ['FLUX_REGION_SEED'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Trainer is launched with its checkout as cwd. Preserve its import resolution.
sys.path.insert(0, os.getcwd())
sys.argv[0] = 'run.py'
runpy.run_path('run.py', run_name='__main__')
