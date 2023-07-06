from configuration import config
import numpy as np
from torch.utils.data import random_split
from utils.preprocess_shift import collate_fn, get_shift_datalist
from utils.data_loader_shift import SHIFTDataset
from utils.train_utils import select_model
from collections import defaultdict
from eval_utils.engine import evaluate
import logging, os, torch
from tqdm import tqdm
from torchvision import transforms
import csv
from torch.utils.tensorboard import SummaryWriter

args = config.joint_parser()

# Setup logging
# Setup logging
log_path = f"logs/{args.dataset}_joint_{'upperbound' if args.upperbound else f'seed_{args.seed_num}'}"
log_path += "_debug.log" if args.debug else ".log"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_path, mode='w'), 
                            logging.StreamHandler()])
task_seed_list = [[0,1,2], [2,0,3],[4,1,3]]

if args.upperbound:
    selected_seed = [0, 1, 2, 3, 4]
    logging.info(f"Starts with no seed (shuffling all tasks)")
else:
    selected_seed = task_seed_list[int(args.seed_num) - 1]
    logging.info(f"Joint training with mixed two tasks")
    logging.info(f"Selected seed: {selected_seed}")

# Set up logging
if args.upperbound:
    tensorboard_pth = os.path.join(args.tensorboard_pth, args.dataset, "joint", f"upperbound")
else:
    tensorboard_pth = os.path.join(args.tensorboard_pth, args.dataset, "joint", f"seed_{args.seed_num}")

writer = SummaryWriter(log_dir=tensorboard_pth)


# Transform definition
transform = transforms.ToTensor()

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

def online_evaluate(model, test_dataloader, sample_num, device):
    coco_evaluator = evaluate(model, test_dataloader, device)
    stats = coco_evaluator.coco_eval['bbox'].stats
    return stats[1]

# # Always test on all tasks regardless of upperbound or not
# domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy']
# test_loader_list = []
# if not args.debug:
#     logging.info("Loading test dataset...")
#     for i in range(len(domain_list)):
#         dataset = SHIFTDataset(task_num=i+1, domain_dict={'weather_coarse': domain_list[i]},
#                                             split="minival", transforms=transforms.ToTensor())
#         test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, collate_fn=collate_fn))
# else:
#     logging.info("Loading test debug dataset...")
#     for i in range(len(domain_list)):
#         test_dataset = SHIFTDataset(task_num=i+1, domain_dict={'weather_coarse': domain_list[i]},
#                                             split="minival", transforms=transforms.ToTensor())
#         debug_dataset, _ = random_split(test_dataset, [10, len(test_dataset) - 10])
#         test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=args.batchsize, collate_fn=collate_fn))

# Training dataset depends on whether it is upperbound or not
# If upperbound, we need to load all tasks, otherwise, we only load the task specified by seed
if not args.debug:
    logging.info("Loading joint dataset for training...")
    logging.info(f"Loading seed {args.seed_num} dataset...")
    logging.info(f"Corresponding task list: {selected_seed}")

    # Load the dataset according to the seed
    joint_dataset = SHIFTDataset(task_num=1, domain_dict=None, split="train")
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batchsize, 
                                                collate_fn=collate_fn, shuffle=True)
    
else:
    # Debug mode
    logging.info("Loading joint debug dataset...")
    logging.info(f"Loading seed {args.seed_num} dataset...")
    logging.info(f"Corresponding task list: {selected_seed}")

    # Load the dataset according to the seed
    debug_joint_dataset = SHIFTDataset(task_num=1, domain_dict=None, split="train")
    joint_dataset, _ = random_split(debug_joint_dataset, [50, len(debug_joint_dataset) - 50])
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batchsize, 
                                                collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = select_model(mode="shift_er", num_classes=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
model.to(device)

samples_cnt = 0
eval_results = defaultdict(list)
epoch_results = defaultdict(list)
eval_count = 1

# Open file to write results (create if not exist, overwrite if exist)
filename_prefix = f"results/{args.dataset}_joint_{'upperbound' if args.upperbound else f'seed_{args.seed_num}'}"
filename_prefix += "_debug" if args.debug else ""

os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

# # Create csv file to write results
# eval_results_file = open(f"{filename_prefix}_eval_results.csv", "w")
# eval_results_writer = csv.writer(eval_results_file)
# eval_results_writer.writerow(["epoch", "test_mAP", "task_evaluating", "data_cnt"])

# Train the model for num_epochs
logging.info("Start training...")
logging.info(f"upperbound: {args.upperbound}, seed_num: {args.seed_num}, num_epochs: {args.num_epochs}")

for ep in range(args.num_epochs):
    logging.info(f"Epoch {ep + 1} / {args.num_epochs}")
    for i, data in enumerate(tqdm(joint_dataloader)):
        samples_cnt += args.batchsize

        # Load the data and send to device
        images = list(transform(img).to(device) for img in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        writer.add_scalar('Loss/train', losses, samples_cnt)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # When the number of samples reaches the evaluation period, evaluate the model
        # if samples_cnt > args.eval_period * eval_count:
        #     eval_count += 1
        #     logging.info(f"Jointly training, upperbound: {args.upperbound}, seed_num: {args.seed_num}, num_epochs: {args.num_epochs}")
        #     logging.info(f"Current Epoch: {ep + 1} / {args.num_epochs}, Step {samples_cnt}, Current Loss: {losses}")

        #     task_mAP_list = []
        #     for task_eval in range(len(domain_list)):
        #         logging.info(f"Epoch {ep + 1} / {args.num_epochs}, Task {task_eval + 1} Evaluation")
        #         mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
        #         task_mAP_list.append(mAP)
        #         eval_results["epoch"].append(ep + 1)
        #         eval_results["test_mAP"].append(mAP)
        #         eval_results["task_evaluating"].append(task_eval + 1)
        #         eval_results["data_cnt"].append(samples_cnt)

        #         # Write the evaluation results to csv file
        #         eval_results_writer.writerow([ep + 1, mAP, task_eval + 1, samples_cnt])
        #         eval_results_file.flush()
        #         os.fsync(eval_results_file.fileno())

        #         writer.add_scalar(f'mAP/task{task_eval + 1}', mAP, samples_cnt)
            
        #     # Write the average mAP of all tasks to tensorboard
        #     average_mAP = sum(task_mAP_list) / float(len(task_mAP_list))
        #     writer.add_scalar("Average mAP", average_mAP, samples_cnt)
        
        #     # Set model to training mode
        #     model.train()

    # After training one epoch is done, save the model
    if args.upperbound:
        save_path = os.path.join("model_checkpoints", args.dataset, "joint", "upperbound")
    else:
        save_path = os.path.join("model_checkpoints", args.dataset, "joint", f"seed_{args.seed_num}")
    
    save_path += "_debug" if args.debug else ""

    logging.info(f"Saving model at epoch {ep + 1}...")

    # If not exist, create the save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{ep + 1}.pth"))

    # After training one epoch is done, starts evaluating each task
    # task_eval_results = []
    # for task_eval in [0, 1, 2, 3]:
    #     mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
    #     task_eval_results.append(mAP)
    # epoch_mAP = sum(task_eval_results) / float(len(task_eval_results))
    # epoch_results["epoch_mAP"].append(epoch_mAP)

# logging.info("Training is done! Saving the evaluation results...")
# # Create the save path if not exist
# if args.upperbound:
#     save_path = os.path.join("outputs", "joint", "upperbound", args.dataset)
# else:
#     save_path = os.path.join("outputs", "joint", args.dataset, f"seed_{args.seed_num}")

# save_path += "_debug" if args.debug else ""

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# Save the evaluation results
# np.save(os.path.join(save_path, "_epoch.npy"), eval_results['epoch'])
# np.save(os.path.join(save_path, "_eval.npy"), eval_results['test_mAP'])
# np.save(os.path.join(save_path, "_eval_task.npy"), eval_results['task_evaluating'])
# np.save(os.path.join(save_path, "_eval_time.npy"), eval_results['data_cnt'])
