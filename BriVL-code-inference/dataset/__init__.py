import torch
from .xybDataset import XYBDataset_all

import os

__all__ = {
    'XYBDataset_all': XYBDataset_all,
}

def build_moco_dataset(args, cfg=None, is_training=True): 
    Dataset = __all__[cfg.DATASET.NAME]

    dataset_val = Dataset(cfg, args, 'val')

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATASET.WORKERS,
        pin_memory=True,
        drop_last=False
    )

    return dataloader_val

