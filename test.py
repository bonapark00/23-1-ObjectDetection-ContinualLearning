from utils.data_loader_shift import SHIFTDataset
from utils.data_loader_clad import get_clad_datalist
from utils.data_loader_shift import get_shift_datalist
from torchvision import transforms
from tqdm import tqdm


cur_train_datalist = get_shift_datalist(data_type="train", task_num=1, domain_dict=None)

list_category_indices = []
for i in range(len(cur_train_datalist)):
    category_ids = cur_train_datalist[i]['objects']['category_id']
    list_category_indices += category_ids
    # Get unique category indices
    list_category_indices = list(set(list_category_indices))

breakpoint()