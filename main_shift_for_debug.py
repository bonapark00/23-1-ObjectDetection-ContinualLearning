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
from utils.preprocess_shift import get_shift_datalist, collate_fn
from utils.data_loader_shift import SHIFTDataset
from utils.method_manager import select_method
from torch.utils import tensorboard

def main():
    args = config.base_parser()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transform definition
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensorboard_path = f"debug_efficiency/debug"
    writer = tensorboard.SummaryWriter(log_dir=f"tensorboard/{tensorboard_path}")
    method = select_method(args, None, device, train_transform, test_transform, 7, writer)

    # 37041, 25433, 13596, 39016, 25966
    domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy']
    # Get train dataset
    if not args.debug:
        print("Loading train dataset...")
        train_task = []
        for i, domain in enumerate(domain_list):
            print(f"Loading task {i+1}...")
            cur_train_datalist = get_shift_datalist(data_type="train", task_num=i+1, domain_dict=
                                                    {'weather_coarse': domain})
            train_task.append(cur_train_datalist)
    else:
        print("Loading train debug dataset...")
        train_task = []
        for i, domain in enumerate(domain_list):
            print(f"Loading task {i+1}...")
            cur_train_datalist = get_shift_datalist(data_type="train", task_num=i+1, domain_dict=
                                                    {'weather_coarse': domain})[:50]
            train_task.append(cur_train_datalist)

    # Get test dataset
    if not args.debug:
        print("Loading test dataset...")
        test_loader_list = []
        for i in range(len(domain_list)):
            print(f"Loading task {i+1}...")
            test_dataset = SHIFTDataset(task_num=i+1, domain_dict={'weather_coarse': domain_list[i]},
                                            split="minival", transforms=transforms.ToTensor())
            test_loader_list.append(torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=collate_fn))
    else:
        print("Loading test debug dataset...")
        test_loader_list = []
        for i in range(len(domain_list)):
            print(f"Loading task {i+1}...")
            test_dataset = SHIFTDataset(task_num=i+1, domain_dict={'weather_coarse': domain_list[i]},
                                            split="minival", transforms=transforms.ToTensor())
            test_dataset = random_split(test_dataset, [50, len(test_dataset) - 50])[0]
            test_loader_list.append(torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=collate_fn))
    
    samples_cnt = 0 # Total number of samples seen
    task_seed_list = [[0,1,2,3,4],[2,0,3,1,4],[4,1,3,0,2]]
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
    # filename_prefix = f"results/{args.dataset}/{args.mode}_{args.batchsize}_{args.temp_batchsize}/seed-{args.seed_num}"
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
        print(f"Mode: {args.mode}, Selected seed: {selected_seed}, Current task: {task + 1} ({domain_list[task]})")
        for j, data in enumerate(tqdm(train_task[task], desc=f"{args.mode} - Seed {args.seed_num} Task {task + 1} ({i+1}/5) ({domain_list[task]}) training")):
            # For each sample, train the model and evaluate when eval period is reached
            samples_cnt += 1
            method.online_step(data, samples_cnt, args.n_worker)
            if samples_cnt % args.eval_period == 0:
                # Evaluate on all tasks before current task
                task_mAP_list = []  # mAP of current task on all tasks before current task
                # TODO: JSON logging would be better
                for task_eval in selected_seed[:i + 1]:
                    print(f"Seed {args.seed_num} Task {task + 1} evaluating task {task_eval + 1}")
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
                    
                    # # Write each task evaluation result to tensorboard
                    # writer.add_scalar(f"task_{task_eval + 1}/mAP", mAP, samples_cnt)
                
                # Write the average mAP of current task on all tasks before current task to tensorboard
                # average_mAP = sum(task_mAP_list) / float(len(task_mAP_list))
                # writer.add_scalar("Average mAP", average_mAP, samples_cnt)
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
            
            print(f"After training task {task + 1}, evaluating task {task_eval + 1}, mAP: {mAP}")

        # Calculate the average mAP of all tasks
        task_mAP = sum(task_records["test_mAP"]) / float(len(task_records["test_mAP"]))
        print(f"After training task {task + 1}, average mAP of all tasks: {task_mAP}")

    # # Create path for each method
    # if not os.path.exists(os.path.join('outputs', args.mode)):
    #     os.makedirs(os.path.join('outputs', args.mode))
    
    # # Create path for after_task
    # if not os.path.exists(os.path.join('outputs', args.mode, 'after_task')):
    #     os.makedirs(os.path.join('outputs', args.mode, 'after_task'))

    # print("Training finished, writing results to file")
    # # Save results to file
    # save_path = (
    #     f"{args.model_name}_{args.dataset}"
    #     f"_bs-{args.batchsize}_tbs-{args.temp_batchsize}"
    #     f"_sd-{args.seed_num}"
    # )

    # # Results during training each task
    # np.save(os.path.join('outputs', args.mode, save_path + "_mAP.npy"), eval_results['test_mAP'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_task_training.npy"), eval_results['task_training'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_task_evaluating.npy"), eval_results['task_evaluating'])
    # np.save(os.path.join('outputs', args.mode, save_path + "_eval_time.npy"), eval_results['data_cnt'])

    # # Results after training each task
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_mAP.npy"), task_records['test_mAP'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_task_trained.npy"), task_records['task_trained'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_task_evaluating.npy"), task_records['task_evaluating'])
    # np.save(os.path.join('outputs', args.mode, "after_task", save_path + "_eval_time.npy"), task_records['data_cnt'])

if __name__ == "__main__":
    main()
