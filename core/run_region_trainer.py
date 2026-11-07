"""Bound text-cache memory and seed RNGs before AI Toolkit's regular CLI."""
import os
import random
import runpy
import sys
from functools import wraps

import numpy as np
import torch

def configure_text_cache():
    from toolkit import dataloader_mixins
    if not hasattr(dataloader_mixins, 'TEXT_EMBED_CACHE_BATCH'):
        raise RuntimeError('unsupported text cache implementation; use the pinned toolkit')
    # The pinned toolkit defaults to 16 prompts per Qwen forward, independently
    # of train.batch_size. That cache batch exhausted a 32 GB RTX 5090.
    dataloader_mixins.TEXT_EMBED_CACHE_BATCH = 1
    item_mixin = dataloader_mixins.TextEmbeddingFileItemDTOMixin
    original_info = item_mixin.get_text_embedding_info_dict
    cache_tag = os.environ['FLUX_REGION_TE_CACHE_TAG']

    @wraps(original_info)
    def precision_aware_info(self, *args, **kwargs):
        info = original_info(self, *args, **kwargs)
        # Upstream hashes captions/embedding-space version but not TE precision.
        # Separate BF16 embeddings from partially populated old FP8 caches.
        info['region_text_encoder_config'] = cache_tag
        return info

    item_mixin.get_text_embedding_info_dict = precision_aware_info
    print('Region trainer: text embedding cache batch = 1', flush=True)
    print(f'Region trainer: text cache configuration = {cache_tag}', flush=True)


def main():
    # Trainer is launched with its checkout as cwd. Preserve its import resolution.
    sys.path.insert(0, os.getcwd())
    configure_text_cache()
    seed = int(os.environ['FLUX_REGION_SEED'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sys.argv[0] = 'run.py'
    runpy.run_path('run.py', run_name='__main__')


if __name__ == '__main__':
    main()
