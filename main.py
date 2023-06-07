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
    
    train_task = [
        cur_train_datalist[0:4470],
        cur_train_datalist[4470:5799],
        cur_train_datalist[5799:7278],
        cur_train_datalist[7278:7802]
    ]

    samples_cnt = 0
    task_seed_list = [[0,1,2,3], [2,0,3,1],[1,2,3,0]]
    
    # Train
    for task in task_seed_list[int(args.seed_num) - 1]:
        for data in tqdm(train_task[task], desc=f"Task {task + 1} / 4"):
            samples_cnt += 1
            method.model.train()
            method.online_step(data, samples_cnt, args.n_worker)
        
        # Save trained model
        save_path = (
            f"{args.mode}_{args.model_name}_{args.dataset}"
            f"_b_size{args.batchsize}_tb_size{args.temp_batchsize}"
            f"_sd_{args.seed_num}_task{task + 1}.pth"
        )
        torch.save(method.model.state_dict(), os.path.join('model_checkpoints', save_path))          


if __name__ == "__main__":
    main()
