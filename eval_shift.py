import torch
import numpy as np
from configuration import config
from torchvision import transforms
from torch.utils.data import random_split
from collections import defaultdict
from tqdm import tqdm
import os

import logging
from utils.preprocess_shift import get_shift_datalist, collate_fn
from utils.data_loader_shift import SHIFTDataset
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
    # Remove existing tensorboard logs
    if os.path.exists(f"tensorboard/{tensorboard_path}"):
        os.system(f"rm -rf tensorboard/{tensorboard_path}")
    writer = tensorboard.SummaryWriter(log_dir=f"tensorboard/{tensorboard_path}")
    method = select_method(args, None, device, train_transform, test_transform, 23, writer)
    # Get train dataset
    domain_dict = {
        'weather_coarse': 'rainy'
    }
    cur_train_datalist = get_shift_datalist(domain_dict=domain_dict, task_num=1, data_type='train')
    # val_data_list = get_shift_datalist('val')
    
    # train_task = [
    #     cur_train_datalist[0:4470],
    #     cur_train_datalist[4470:5799],
    #     cur_train_datalist[5799:7278],
    #     cur_train_datalist[7278:7802]
    # ]
    # if args.debug:
    #     train_task = [
    #         cur_train_datalist[0:20],
    #         cur_train_datalist[4470:4500],
    #         cur_train_datalist[5799:5830],
    #         cur_train_datalist[7278:7300]
    #     ]

    # Get test dataset
    # if not args.debug:
    #     test_loader_list = []
    #     for i in range(4): 
    #         dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
    #                                     split="val", transforms=transforms.ToTensor())

    #         test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn))
    # else:
    #     test_loader_list = []
    #     for i in range(4): 
    #         dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
    #                                     split="val", transforms=transforms.ToTensor())
    #         debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
    #         test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=4, collate_fn=collate_fn))

    # dataset = SHIFTDataset(split="val", transforms=transforms.ToTensor())
    # debug_dataset, _ = random_split(dataset, [500, len(dataset) - 500])
    # shift_dataloader = torch.utils.data.DataLoader(debug_dataset, batch_size=1, collate_fn=collate_fn)

    if not args.debug:
        print("Loading test dataset...")
        test_loader_list = []
        for i in range(4):
            dataset = SHIFTDataset(split="val", transforms=transforms.ToTensor())
            debug_dataset, _ = random_split(dataset, [500, len(dataset) - 500])
            shift_dataloader = torch.utils.data.DataLoader(debug_dataset, batch_size=args.batchsize, collate_fn=collate_fn)

    else:
        print("Loading test debug dataset...")
        test_loader_list = []
        for i in range(4): 
            dataset = SHIFTDataset(split="val", transforms=transforms.ToTensor())
            debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
            shift_dataloader = torch.utils.data.DataLoader(debug_dataset, batch_size=args.batchsize, collate_fn=collate_fn)

    samples_cnt = 0
    # task_seed_list = [[0,1,2,3], [2,0,3,1],[1,2,3,0]]
    eval_results = defaultdict(list)
    task_records = defaultdict(list)
 
    for data in tqdm(cur_train_datalist, desc=f"{args.mode} - Seed {args.seed_num}  training"):
        samples_cnt += 1
        method.model.train()
        method.online_step(data, samples_cnt, args.n_worker)

        if samples_cnt % args.eval_period == 0:
            mAP= method.online_evaluate(shift_dataloader, samples_cnt)
            eval_results["test_mAP"].append(mAP)
            eval_results["task_training"].append(1)
            eval_results["task_eval"].append(1)
            eval_results["data_cnt"].append(samples_cnt)

            writer.add_scalar(f"mAP", mAP, samples_cnt)

    task_eval_results = []
    # breakpoint()
    mAP = method.online_evaluate(shift_dataloader, samples_cnt)
    task_eval_results.append(mAP)

    # task_mAP = sum(task_eval_results) / float(len(task_eval_results))
    # task_records["task_mAP"].append(task_mAP)

    # for data in cur_train_datalist:
    #     print(data)
    #     breakpoint()
        
    # Save results

    logging.info("Training finished! Saving results...")
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
