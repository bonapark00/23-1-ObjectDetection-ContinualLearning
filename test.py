from utils.data_loader_shift import SHIFTDataset
from utils.data_loader_clad import get_clad_datalist
from utils.data_loader_shift import get_shift_datalist
from torchvision import transforms
from tqdm import tqdm
import os
from utils.train_utils import select_pq_dataset


memory = select_pq_dataset(310, [1,2], 'clad')
