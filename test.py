from utils.data_loader_shift import SHIFTDataset
from utils.data_loader_clad import get_clad_datalist
from utils.data_loader_shift import get_shift_datalist
from torchvision import transforms
from tqdm import tqdm
import os

# count the number of images under directory hello
def count_files(path):
    list_hel = os.listdir(path) # dir is your directory path
    number_files = len(list_hel)
    return number_files

# count the number of images under directory hello
def count_files(path):
    list_hel = os.listdir(path) # dir is your directory path
    number_files = len(list_hel)
    return number_files

path = './precomputed_proposals/ssl_shift'
hello = count_files(path)
breakpoint()
