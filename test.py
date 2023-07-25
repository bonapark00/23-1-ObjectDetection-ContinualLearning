from utils.data_loader_shift import SHIFTDataset
from utils.data_loader_clad import get_clad_datalist
from utils.data_loader_shift import get_shift_datalist
from torchvision import transforms
from tqdm import tqdm
import os
from utils.train_utils import select_pq_dataset
from utils.preprocess_clad import get_clad_datalist
import h5py



domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy']
root = './dataset'
train_data_list= []
for i, domain in enumerate(domain_list):
    cur_train_datalist = get_shift_datalist(data_type="train", task_num=i+1, domain_dict=
                                            {'weather_coarse': domain}, root=root)
    train_data_list.extend(cur_train_datalist)
    break

breakpoint()