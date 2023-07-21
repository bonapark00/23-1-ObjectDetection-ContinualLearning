from utils.data_loader_shift import SHIFTDataset
from utils.data_loader_clad import get_clad_datalist
from utils.data_loader_shift import get_shift_datalist
from torchvision import transforms
from tqdm import tqdm
import os
from utils.train_utils import select_pq_dataset
from utils.preprocess_clad import get_clad_datalist
import h5py

path = './rodeo_feature/clad_reconstructed_train_features.h5'
file = h5py.File(path, 'r')
keys = list(file.keys())

breakpoint()



