from configuration import config
import numpy as np
from torch.utils.data import random_split
from utils.preprocess_clad import collate_fn
from utils.data_loader_clad import SODADataset
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
note_suffix = f"_{args.note}" if args.note else ""
log_path = f"logs/{args.dataset}/joint/{'upperbound' if args.upperbound else f'seed_{args.seed_num}'}{note_suffix}.log"

# Create log directory if not exist
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_path, mode='w'), 
                            logging.StreamHandler()])

task_seed_list = [[0,1], [2,0],[1,2]]

if args.upperbound:
    selected_seed = [0, 1, 2, 3]
    logging.info(f"Starts with no seed (shuffling all tasks)")
else:
    selected_seed = task_seed_list[int(args.seed_num) - 1]
    logging.info(f"Joint training with mixed two tasks")
    logging.info(f"Selected seed: {selected_seed}")

# Set up tensorboard logging
tensorboard_path = f"tensorboard/{args.dataset}/joint/{'upperbound' if args.upperbound else f'seed_{args.seed_num}'}{note_suffix}"
# Remove existing tensorboard logs
if os.path.exists(f"tensorboard/{tensorboard_path}"):
    os.system(f"rm -rf tensorboard/{tensorboard_path}")
writer = SummaryWriter(log_dir=tensorboard_path)

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

# Always test on all tasks regardless of upperbound or not
domain_list = ['T1', 'T2', 'T3', 'T4']
test_loader_list = []
if not args.debug:
    logging.info("Loading test dataset...")
    for i in range(len(domain_list)):
        dataset = SODADataset(root=args.dataset_root, task_ids=[i+1],
                                    split="val", transforms=transforms.ToTensor())
        test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn))
else:
    logging.info("Loading test debug dataset...")
    for i in range(len(domain_list)):
        dataset = SODADataset(root=args.dataset_root, task_ids=[i+1],
                                    split="val", transforms=transforms.ToTensor())
        debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
        test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=4, collate_fn=collate_fn))

# Training dataset depends on whether it is upperbound or not
# If upperbound, we need to load all tasks, otherwise, we only load the task specified by seed
if not args.debug:
    logging.info("Loading joint dataset for training...")
    logging.info(f"Loading seed {args.seed_num} dataset...")
    logging.info(f"Corresponding task list: {selected_seed}")

    # Load the dataset according to the seed
    joint_dataset = SODADataset(root=args.dataset_root, task_ids=selected_seed, split="train")
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batchsize, 
                                                collate_fn=collate_fn, shuffle=True)
else:
    # Debug mode
    logging.info("Loading joint debug dataset...")
    logging.info(f"Loading seed {args.seed_num} dataset...")
    logging.info(f"Corresponding task list: {selected_seed}")

    # Load the dataset according to the seed
    debug_joint_dataset = SODADataset(root=args.dataset_root, task_ids=selected_seed, split="train")
    joint_dataset, _ = random_split(debug_joint_dataset, [50, len(debug_joint_dataset) - 50])
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batchsize, 
                                                collate_fn=collate_fn, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = select_model("joint", num_classes=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
model.to(device)

samples_cnt = 0
eval_results = defaultdict(list)
epoch_results = defaultdict(list)
eval_count = 1

# Train the model for num_epochs
logging.info("Start training...")
logging.info(f"upperbound: {args.upperbound}, seed_num: {args.seed_num}, num_epochs: {args.num_epochs}")

for ep in range(args.num_epochs):
    logging.info(f"Epoch {ep + 1} / {args.num_epochs}")
    for i, data in enumerate(tqdm(joint_dataloader)):
        samples_cnt += args.batchsize

        # Load the data and send to device
        # Send images to device
        images = list(transform(img).to(device) for img in data[0])

        # Send targets to device except for the img paths
        targets = [{k: v.to(device) for k, v in t.items() if k != 'img_path'} for t in data[1]]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

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
        #     for task_eval in [0, 1, 2, 3]:
        #         logging.info(f"Epoch {ep + 1} / {args.num_epochs}, Task {task_eval + 1} Evaluation")
        #         mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
        #         task_mAP_list.append(mAP)
        #         eval_results["epoch"].append(ep + 1)
        #         eval_results["test_mAP"].append(mAP)
        #         eval_results["task_evaluating"].append(task_eval + 1)
        #         eval_results["data_cnt"].append(samples_cnt)
        #         writer.add_scalar(f'mAP/task{task_eval + 1}', mAP, samples_cnt)
            
        #     # Write the average mAP of all tasks to tensorboard
        #     average_mAP = sum(task_mAP_list) / float(len(task_mAP_list))
        #     writer.add_scalar("Average mAP", average_mAP, samples_cnt)

    # After training one epoch is done, save the model
    save_path = f"model_checkpoints/{args.dataset}/joint/{'upperbound' if args.upperbound else f'seed_{args.seed_num}'}{note_suffix}"
    logging.info(f"Saving model at epoch {ep + 1}...")

    # If not exist, create the save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{ep + 1}.pth"))

    # After training one epoch is done, starts evaluating each task
    task_eval_results = []
    for task_eval in [0, 1, 2, 3]:
        mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
        task_eval_results.append(mAP)
        epoch_results["epoch"].append(ep + 1)
        epoch_results["test_mAP"].append(mAP)
        epoch_results["task_evaluating"].append(task_eval + 1)

    epoch_mAP = sum(task_eval_results) / float(len(task_eval_results))
    epoch_results["epoch_mAP"].append(epoch_mAP)

    # Set model to train mode
    model.train()

# Create the save path if not exist
save_path = f"outputs/{args.dataset}/joint/{'upperbound' if args.upperbound else f'seed_{args.seed_num}'}{note_suffix}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# # Save the evaluation results
# np.save(os.path.join(save_path, "epoch.npy"), eval_results['epoch'])
# np.save(os.path.join(save_path, "eval.npy"), eval_results['test_mAP'])
# np.save(os.path.join(save_path, "eval_task.npy"), eval_results['task_evaluating'])
# np.save(os.path.join(save_path, "eval_time.npy"), eval_results['data_cnt'])

# Save epoch results
np.save(os.path.join(save_path, "epoch.npy"), epoch_results['epoch'])
np.save(os.path.join(save_path, "mAP.npy"), epoch_results['test_mAP'])
np.save(os.path.join(save_path, "eval_task.npy"), epoch_results['task_evaluating'])
np.save(os.path.join(save_path, "epoch_mAP.npy"), epoch_results['epoch_mAP'])

