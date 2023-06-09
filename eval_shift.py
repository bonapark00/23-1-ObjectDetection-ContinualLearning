import torch
import os
from torchvision import transforms
from utils.data_loader_shift import SHIFTDataset
from utils.method_manager import select_method
from utils.preprocess_clad import collate_fn
from configuration import config
from eval_utils.engine import evaluate

data_set=SHIFTDataset()
shift_dataloader = torch.utils.data.DataLoader(data_set, batch_size=1, collate_fn=collate_fn)


for sample in shift_dataloader:
    print(sample)
    breakpoint()