import torch
from configuration import config
from torchvision import transforms
import os
from tqdm import tqdm
import logging
from utils.preprocess_clad import get_clad_datalist
from utils.method_manager import select_method


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
    
    
    method = select_method(args, None, device, train_transform, test_transform, 7)
    
    cur_train_datalist = get_clad_datalist('train')
    # cur_test_datalist = get_clad_datalist('val')
    
    train_task = [
        cur_train_datalist[0:4470],
        cur_train_datalist[4470:5799],
        cur_train_datalist[5799:7278],
        cur_train_datalist[7278:7802]
    ]
    
    # task_1_val = cur_test_datalist[0:4470]
    # task_2_val= cur_test_datalist[4470:5799]
    # task_3_val = cur_test_datalist[5799:7278]
    # task_4_val = cur_test_datalist[7278:7802]
    
    # val_task = [task_1_val, task_2_val, task_3_val, task_4_val]
    
    # train_num = [0, 4470, 5799, 7278, 7802]
    # val_num = [0, 497, 645, 810, 869]

    samples_cnt = 0
    # test_loader_list = []
    
    # for i in range(4): 
    #     data_set = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
    #                               split="val", transforms=data_transform)
    
    #     test_loader_list.append(torch.utils.data.DataLoader(data_set, batch_size=4, collate_fn=collate_fn))
    
    # Changed seed order to check training status
    task_seed_list = [[2,0,3,1],[0,1,2,3],[1,2,3,0]]
    
    # Train
    for task in task_seed_list[args.seed_num]:
    
    for i in range(3):
        for task in task_list[i]:
            for data in tqdm(train_task[task], desc=f"Task {task+1}/4"):
                samples_cnt += 1
                method.model.train()
                method.online_step(data, samples_cnt, args.n_worker)
            
            # method.model.eval()
            # evaluate(method.model, test_loader_list[task], device=device)
            torch.save(method.model.state_dict(), os.path.join('model_checkpoints', f"clad_der_model_task_{task+1}_seed{seed_num[i]}.pth"))
        method = select_method(args, None, device, train_transform, test_transform, 7)

if __name__ == "__main__":
    main()
