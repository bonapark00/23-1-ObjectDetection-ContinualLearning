import torch
import numpy as np
from configuration import config
from torchvision import transforms
from torch.utils.data import random_split
from collections import defaultdict
from tqdm import tqdm
import csv
import os

import logging
from utils.preprocess_clad import get_clad_datalist, collate_fn
from utils.data_loader_clad import SODADataset
from utils.method_manager import select_method
from torch.utils import tensorboard
from calculate_auc import get_mAP_AUC

def main():
    args = config.base_parser()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    note_suffix = f"_{args.note}" if args.note else ""
    log_path = f"logs/{args.dataset}/{args.mode}/sd-{args.seed_num}{note_suffix}.log"

    # Create log directory if not exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w'), 
                                logging.StreamHandler()])

    # Show arguments
    logging.info("Visualizing arguments...")
    logging.info(f"mode: {args.mode}")
    logging.info(f"dataset: {args.dataset}")
    logging.info(f"eval period: {args.eval_period}")
    logging.info(f"memory size: {args.memory_size}")
    logging.info(f"batch size: {args.batch_size}")
    logging.info(f"temp_batch size: {args.temp_batch_size}")
    logging.info(f"seed num: {args.seed_num}")
    logging.info(f"note: {args.note}")

    save_path = "model_checkpoints"
    os.makedirs(save_path, exist_ok=True)

    # Transform definition
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    tensorboard_path = f"{args.dataset}_{args.mode}_sd-{args.seed_num}{note_suffix}"
    # Remove existing tensorboard logs
    if os.path.exists(f"tensorboard/{tensorboard_path}"):
        os.system(f"rm -rf tensorboard/{tensorboard_path}")
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
        logging.info("Loading test dataset...")
        test_loader_list = []
        for i in range(4):
            dataset = SODADataset(path="./dataset/SSLAD-2D", task_ids=[i+1],
                                        split="val", transforms=transforms.ToTensor())

            test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, collate_fn=collate_fn))

    else:
        logging.info("Loading test debug dataset...")
        test_loader_list = []
        for i in range(4): 
            dataset = SODADataset(path="./dataset/SSLAD-2D", task_ids=[i+1],
                                        split="val", transforms=transforms.ToTensor())
            debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
            test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=args.batchsize, collate_fn=collate_fn))

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

    # # Open file to write results (create if not exist, overwrite if exist)
    # filename_prefix = f"results/{args.dataset}/{args.mode}/seed-{args.seed_num}{note_suffix}"
    # os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

    # # Create csv files to write results
    # eval_results_file = open(f"{filename_prefix}_eval_results.csv", "w")
    # eval_results_writer = csv.writer(eval_results_file)
    # eval_results_writer.writerow(["test_mAP", "task_training", "task_evaluating", "data_cnt"])

    # task_records_file = open(f"{filename_prefix}_task_records.csv", "w")
    # task_records_writer = csv.writer(task_records_file)
    # task_records_writer.writerow(["test_mAP", "task_trained", "task_evaluating", "data_cnt"])

    # Train and eval
    for i, task in enumerate(selected_seed):
        # Train one task
        method.model.train()
        logging.info(f"Mode: {args.mode}, Selected seed: {selected_seed}, Current task: {task + 1}")
        for data in tqdm(train_task[task], desc=f"{args.mode} - Seed {args.seed_num} Task {task + 1} ({i+1}/4) training"):
            # For each sample, train the model and evaluate when eval period is reached
            samples_cnt += 1
            method.online_step(data, samples_cnt, args.n_worker)
            if samples_cnt % args.eval_period == 0:
                # Evaluate on all tasks before current task
                task_mAP_list = []  # mAP of current task on all tasks before current task
                # TODO: JSON logging would be better
                for task_eval in selected_seed[:i + 1]:
                    logging.info(f"Seed {args.seed_num} Task {task + 1} evaluating task {task_eval + 1}")
                    mAP = method.online_evaluate(test_loader_list[task_eval], samples_cnt)
                    task_mAP_list.append(mAP)
                    eval_results["test_mAP"].append(mAP)
                    eval_results["task_training"].append(task + 1)
                    eval_results["task_evaluating"].append(task_eval + 1)
                    eval_results["data_cnt"].append(samples_cnt)

                    # # Write current evaluation result to csv file
                    # eval_results_writer.writerow([mAP, task + 1, task_eval + 1, samples_cnt])
                    # eval_results_file.flush()
                    # os.fsync(eval_results_file.fileno())
                    
                    # Write each task evaluation result to tensorboard
                    writer.add_scalar(f"task_{task_eval + 1}/mAP", mAP, samples_cnt)
                
                # Write the average mAP of current task on all tasks before current task to tensorboard
                average_mAP = sum(task_mAP_list) / float(len(task_mAP_list))
                writer.add_scalar("Average mAP", average_mAP, samples_cnt)
                method.model.train()

        # After training one task, evaluate on all tasks including before and after current task
        for task_eval in selected_seed:
            mAP = method.online_evaluate(test_loader_list[task_eval], samples_cnt)
            task_records["test_mAP"].append(mAP)
            task_records["task_trained"].append(task + 1)
            task_records["task_evaluating"].append(task_eval + 1)
            task_records["data_cnt"].append(samples_cnt)

            # # Write current evaluation result to csv file
            # task_records_writer.writerow([mAP, task + 1, task_eval + 1, samples_cnt])
            # task_records_file.flush()
            # os.fsync(task_records_file.fileno())
            
            logging.info(f"After training task {task + 1}, evaluating task {task_eval + 1}, mAP: {mAP}")

        # Calculate the average mAP of all tasks
        task_mAP = sum(task_records["test_mAP"]) / float(len(task_records["test_mAP"]))
        logging.info(f"After training task {task + 1}, average mAP of all tasks: {task_mAP}")

    logging.info("Training finished, writing results to file")
    # Save results to file
    save_path = os.path.join('outputs', args.dataset, args.mode, f"sd-{args.seed_num}{note_suffix}")

    # Create path for each seed
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "after_task"), exist_ok=True)

    # Results during training each task
    np.save(os.path.join(save_path, "mAP.npy"), eval_results['test_mAP'])
    np.save(os.path.join(save_path, "task_training.npy"), eval_results['task_training'])
    np.save(os.path.join(save_path, "task_evaluating.npy"), eval_results['task_evaluating'])
    np.save(os.path.join(save_path, "eval_time.npy"), eval_results['data_cnt'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_mAP.npy"), eval_results['test_mAP'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_task_training.npy"), eval_results['task_training'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_task_evaluating.npy"), eval_results['task_evaluating'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_eval_time.npy"), eval_results['data_cnt'])

    # Results after training each task
    np.save(os.path.join(save_path, "after_task", "mAP.npy"), task_records['test_mAP'])
    np.save(os.path.join(save_path, "after_task", "task_trained.npy"), task_records['task_trained'])
    np.save(os.path.join(save_path, "after_task", "task_evaluating.npy"), task_records['task_evaluating'])
    np.save(os.path.join(save_path, "after_task", "eval_time.npy"), task_records['data_cnt'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_mAP.npy"), task_records['test_mAP'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_task_trained.npy"), task_records['task_trained'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_task_evaluating.npy"), task_records['task_evaluating'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_eval_time.npy"), task_records['data_cnt'])

    # Get 4 x 4 matrix using after task mAP
    after_mAP = np.array(task_records['test_mAP']).reshape(4, 4).T

    # Save as csv
    # Make row and column names
    csv_save_path = os.path.join('outputs', 'summary', f"{args.dataset}_{args.mode}_sd-{args.seed_num}{note_suffix}.csv")
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)

    col_names = ["Task 1", "Task 2", "Task 3", "Task 4"]

    # Get AUC information
    # First, convert lists in task_records to numpy arrays
    for key in task_records:
        task_records[key] = np.array(task_records[key])

    auc_dict = get_mAP_AUC(task_records, 4)
    mAP_AUC = auc_dict["mAP_AUC"]
    logging.info(f"mAP AUC: {mAP_AUC}")
    last_auc_row = np.array([[mAP_AUC, mAP_AUC, mAP_AUC, mAP_AUC]])
    after_mAP = np.append(after_mAP, last_auc_row, axis=0)

    np.savetxt(csv_save_path, after_mAP, delimiter=",", header=",".join(col_names), comments="", fmt="%s")

if __name__ == "__main__":
    main()
