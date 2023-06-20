import torch
import numpy as np
from configuration import config
from torchvision import transforms
from torch.utils.data import random_split
from collections import defaultdict
import os
from tqdm import tqdm
from itertools import islice

import logging
from utils.preprocess_clad import get_clad_datalist, collate_fn
from utils.data_loader_clad import SODADataset
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
    
    # Get train dataset
    cur_train_datalist = get_clad_datalist('train')
    
    train_task = [
        cur_train_datalist[0:4470],
        cur_train_datalist[4470:5799],
        cur_train_datalist[5799:7278],
        cur_train_datalist[7278:7802]
    ]
    if args.debug:
        train_task = [
            cur_train_datalist[0:20],
            cur_train_datalist[4470:4500],
            cur_train_datalist[5799:5830],
            cur_train_datalist[7278:7300]
        ]

    # Get test dataset
    if not args.debug:
        test_loader_list = []
        for i in range(4): 
            dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                        split="val", transforms=transforms.ToTensor())

            test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn))
    else:
        test_loader_list = []
        for i in range(4): 
            dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                        split="val", transforms=transforms.ToTensor())
            debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
            test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=4, collate_fn=collate_fn))

    samples_cnt = 0
    task_seed_list = [[0,1,2,3], [2,0,3,1],[1,2,3,0]]
    eval_results = defaultdict(list)
    task_records = defaultdict(list)
    selected_seed = task_seed_list[int(args.seed_num) - 1]

    # Train and eval
    for i, task in enumerate(selected_seed):
        for data in train_task[task]:
            samples_cnt += 1
            method.model.train()
            method.online_step(data, samples_cnt, args.n_worker)
            if samples_cnt % args.eval_period == 0:
                for task_eval in selected_seed[:i + 1]:
                    breakpoint()
                    mAP = method.online_evaluate(test_loader_list[task_eval], samples_cnt)
                    eval_results["test_mAP"].append(mAP)
                    eval_results["task_training"].append(task + 1)
                    eval_results["task_eval"].append(task_eval + 1)
                    eval_results["data_cnt"].append(samples_cnt)

        # Training one task is done, starts evaluating each task
        # TODO: After training one task, should we evaluate all tasks?
        task_eval_results = []
        for task_eval in selected_seed[:i + 1]:
            mAP = method.online_evaluate(test_loader_list[task_eval], samples_cnt)
            task_eval_results.append(mAP)
        task_mAP = sum(task_eval_results) / float(len(task_eval_results))
        task_records["task_mAP"].append(task_mAP)
        
    # Save results
    save_path = (
        f"{args.mode}_{args.model_name}_{args.dataset}"
        f"_b_size{args.batchsize}_tb_size{args.temp_batchsize}"
        f"_sd_{args.seed_num}"
    )
    np.save(os.path.join('outputs', save_path + ".npy"), task_records["task_mAP"])
    np.save(os.path.join('outputs', save_path + "_eval.npy"), eval_results['test_mAP'])
    np.save(os.path.join('outputs', save_path + "_training_task.npy"), eval_results['task_training'])
    np.save(os.path.join('outputs', save_path + "eval_task.npy"), eval_results['task_eval'])
    np.save(os.path.join('outputs', save_path + "_eval_time.npy"), eval_results['data_cnt'])
    
        # # Save trained model
        # save_path = (
        #     f"{args.mode}_{args.model_name}_{args.dataset}"
        #     f"_b_size{args.batchsize}_tb_size{args.temp_batchsize}"
        #     f"_sd_{args.seed_num}_task{task + 1}.pth"
        # )
        # torch.save(method.model.state_dict(), os.path.join('model_checkpoints', save_path))


if __name__ == "__main__":
    main()
