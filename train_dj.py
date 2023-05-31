import torch
from torch.utils.data import DataLoader
from soda import SODADataset
from configuration import config
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
from tqdm import tqdm
import logging
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from clad_utils import collate_fn, data_transform, get_model_instance_segmentation
from methods.filod import FILOD_DJ
from methods.clad_der import CLAD_DER
from clad_data import get_clad_datalist


def main():
    args = config.base_parser() 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('training.log', mode='w'), 
                                logging.StreamHandler()])
    save_path = "model_checkpoints"
    os.makedirs(save_path, exist_ok=True)

    # Transform definition
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    
    batch_size = 4
    method =CLAD_DER(None, device, train_transform, test_transform, 7, **vars(args))
    
    cur_train_datalist = get_clad_datalist('train')
    cur_test_datalist = get_clad_datalist('val')
    
    task_1_train = cur_train_datalist[0:4470]
    task_2_train = cur_train_datalist[4470:5799]
    task_3_train = cur_train_datalist[5799:7278]
    task_4_train = cur_train_datalist[7278:7802]
    
    train_task = [task_1_train, task_2_train, task_3_train, task_4_train]
    
    task_1_val = cur_test_datalist[0:4470]
    task_2_val= cur_test_datalist[4470:5799]
    task_3_val = cur_test_datalist[5799:7278]
    task_4_val = cur_test_datalist[7278:7802]
    
    val_task = [task_1_val, task_2_val, task_3_val, task_4_val]
    
    train_num = [0, 4470, 5799, 7278, 7802]
    val_num = [0, 497, 645, 810, 869]

    samples_cnt = 0
    test_loader_list = []
    
    for i in range(4): 
        data_set = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                  split="val", transforms=data_transform)
    
        test_loader_list.append(torch.utils.data.DataLoader(data_set, batch_size=4, collate_fn=collate_fn))
    
    #changed seed order to check training status
    task_list = [[2,0,3,1],[0,1,2,3],[1,2,3,0]]
    seed_num = [2,1,3]

    for i in range(3):
        for task in task_list[i]:
            method.seed_num = seed_num[i]
            for data in tqdm(train_task[task], desc=f"Task {task+1}/4"):
                samples_cnt += 1
                method.model.train()
                method.online_step(data, samples_cnt, args.n_worker)
            
            
            #method.model.eval()
            #evaluate(method.model, test_loader_list[task], device=device)
            torch.save(method.model.state_dict(), os.path.join('model_checkpoints', f"clad_der_model_task_{task+1}_seed{seed_num[i]}.pth"))
        method =CLAD_DER(None, device, train_transform, test_transform, 7, **vars(args))
if __name__ == "__main__":
    main()
    
    
    
     