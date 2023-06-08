import torch
import numpy as np
from configuration import config
from torchvision import transforms
from torch.utils.data import random_split
from collections import defaultdict
import os

import logging
from utils.preprocess_clad import get_clad_datalist, collate_fn
from utils.data_loader_clad import SODADataset
from utils.method_manager import select_method
from torch.utils import tensorboard

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
    
    tensorboard_path = f"{args.mode}_{args.model_name}_{args.dataset}_bs-{args.batchsize}_tbs-{args.temp_batchsize}_sd-{args.seed_num}"
    writer = tensorboard.SummaryWriter(log_dir=f"tensorboard/{tensorboard_path}")
    method = select_method(args, None, device, train_transform, test_transform, 7, writer)
    
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
        print("Loading test dataset...")
        test_loader_list = []
        for i in range(4):
            dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                        split="val", transforms=transforms.ToTensor())

            test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn))

    else:
        print("Loading test debug dataset...")
        test_loader_list = []
        for i in range(4): 
            dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                        split="val", transforms=transforms.ToTensor())
            debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
            test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=4, collate_fn=collate_fn))

    samples_cnt = 0 # Total number of samples seen
    task_seed_list = [[0,1,2,3], [2,0,3,1],[1,2,3,0]]
    eval_results = defaultdict(list)    # Evaluation results (any time inference, each eval period, only tasks before current task)
    task_records = defaultdict(list)    # Evaluation results after training each task
    selected_seed = task_seed_list[int(args.seed_num) - 1]

    """
        What we need to log:
        1. When eval period is reached, evaluate on all tasks before current task
            - Log the (test_mAP, task_training, task_evaluating, data_cnt) in eval_results dictionary
            - test_mAP: mAP of current task on all tasks before current task
            - task_training: current task
            - task_evaluating: currently evaluating task
            - data_cnt: number of samples seen
        
        2. After training one task, evaluate on all tasks including before and after current task in task_records dictionary
            - Log the (test_mAP, task_trained, task_evaluating, data_cnt) in task_records dictionary
        
        3. Tensorboard logging
            - Tensorboard logging is done by passing writer to method constructor
            - In each method, online_step and online_evaluate should have writer as input
    """

    # Train and eval
    for i, task in enumerate(selected_seed):
        # Train one task
        logging.info(f"Mode: {args.mode}, Selected seed: {selected_seed}, Current task: {task + 1}")
        for data in train_task[task]:
            # For each sample, train the model and evaluate when eval period is reached
            samples_cnt += 1
            method.model.train()
            loss = method.online_step(data, samples_cnt, args.n_worker)
            logging.info(f"Task {task + 1}, sample {samples_cnt}, loss: {loss}")
            
            if samples_cnt % args.eval_period == 0:
                # Evaluate on all tasks before current task
                for task_eval in selected_seed[:i + 1]:
                    mAP = method.online_evaluate(test_loader_list[task_eval], samples_cnt)
                    eval_results["test_mAP"].append(mAP)
                    eval_results["task_training"].append(task + 1)
                    eval_results["task_evaluating"].append(task_eval + 1)
                    eval_results["data_cnt"].append(samples_cnt)

        # After training one task, evaluate on all tasks including before and after current task
        for task_eval in selected_seed:
            mAP = method.online_evaluate(test_loader_list[task_eval], samples_cnt)
            task_records["test_mAP"].append(mAP)
            task_records["task_trained"].append(task + 1)
            task_records["task_evaluating"].append(task_eval + 1)
            task_records["data_cnt"].append(samples_cnt)

            logging.info(f"After training task {task + 1}, evaluating task {task_eval + 1}, mAP: {mAP}")

        # Calculate the average mAP of all tasks
        task_mAP = sum(task_records["test_mAP"]) / float(len(task_records["test_mAP"]))
        logging.info(f"After training task {task + 1}, average mAP of all tasks: {task_mAP}")

    # Save results to file
    save_path = (
        f"{args.mode}_{args.model_name}_{args.dataset}"
        f"_bs-{args.batchsize}_tbs-{args.temp_batchsize}"
        f"_sd-{args.seed_num}"
    )

    # Results during training each task
    np.save(os.path.join('outputs', save_path + "_mAP.npy"), eval_results['test_mAP'])
    np.save(os.path.join('outputs', save_path + "_task_training.npy"), eval_results['task_training'])
    np.save(os.path.join('outputs', save_path + "_task_evaluating.npy"), eval_results['task_evaluating'])
    np.save(os.path.join('outputs', save_path + "_eval_time.npy"), eval_results['data_cnt'])

    # Results after training each task
    np.save(os.path.join('outputs', save_path + "_mAP.npy"), task_records['test_mAP'])
    np.save(os.path.join('outputs', save_path + "_task_trained.npy"), task_records['task_trained'])
    np.save(os.path.join('outputs', save_path + "_task_evaluating.npy"), task_records['task_evaluating'])
    np.save(os.path.join('outputs', save_path + "_eval_time.npy"), task_records['data_cnt'])

if __name__ == "__main__":
    main()
