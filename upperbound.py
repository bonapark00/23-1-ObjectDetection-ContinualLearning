from configuration import config
import numpy as np
from torch.utils.data import random_split
from utils.preprocess_clad import collate_fn
from utils.data_loader_clad import SODADataset
from utils.train_utils import select_model
from collections import defaultdict
from eval_utils.engine import evaluate
from utils.visualize import visualize_bbox
import logging, os, torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log', mode='w'), 
                            logging.StreamHandler()])

args = config.joint_parser()
task_seed_list = [[0,1], [2,0],[1,2]]

if args.upperbound:
    selected_seed = [0, 1, 2, 3]
    logging.info(f"Starts with no seed (shuffling all tasks)")
else:
    selected_seed = task_seed_list[int(args.seed_num) - 1]
    logging.info(f"Joint training with mixed two tasks")
    logging.info(f"Selected seed: {selected_seed}")

# Set up logging
if args.upperbound:
    tensorboard_pth = os.path.join(args.tensorboard_pth, "joint", f"upperbound")
else:
    tensorboard_pth = os.path.join(args.tensorboard_pth, "joint", f"seed_{args.seed_num}")

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

# Always test on all tasks regardless of upperbound or not
test_loader_list = []
if not args.debug:
    logging.info("Loading test dataset...")
    for i in range(4):
        dataset = SODADataset(path="./dataset/SSLAD-2D", task_ids=[i+1],
                                    split="val", transforms=transforms.ToTensor())
        test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn))
else:
    logging.info("Loading test debug dataset...")
    for i in range(4):
        dataset = SODADataset(path="./dataset/SSLAD-2D", task_ids=[i+1],
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
    joint_dataset = SODADataset(path="./dataset/SSLAD-2D", task_ids=selected_seed, split="train")
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batch_size, 
                                                collate_fn=collate_fn, shuffle=True)
else:
    # Debug mode
    logging.info("Loading joint debug dataset...")
    logging.info(f"Loading seed {args.seed_num} dataset...")
    logging.info(f"Corresponding task list: {selected_seed}")

    # Load the dataset according to the seed
    debug_joint_dataset = SODADataset(path="./dataset/SSLAD-2D", task_ids=selected_seed, split="train")
    joint_dataset, _ = random_split(debug_joint_dataset, [50, len(debug_joint_dataset) - 50])
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batch_size, 
                                                collate_fn=collate_fn, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = select_model("frcnn", None, 7, False)
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
        model.train()
        samples_cnt += args.batch_size

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
        if samples_cnt > args.eval_period * eval_count:
            eval_count += 1
            logging.info(f"Jointly training, upperbound: {args.upperbound}, seed_num: {args.seed_num}, num_epochs: {args.num_epochs}")
            logging.info(f"Current Epoch: {ep + 1} / {args.num_epochs}, Step {samples_cnt}, Current Loss: {losses}")

            task_mAP_list = []
            for task_eval in [0, 1, 2, 3]:
                logging.info(f"Epoch {ep + 1} / {args.num_epochs}, Task {task_eval + 1} Evaluation")
                mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
                task_mAP_list.append(mAP)
                eval_results["epoch"].append(ep + 1)
                eval_results["test_mAP"].append(mAP)
                eval_results["task_evaluating"].append(task_eval + 1)
                eval_results["data_cnt"].append(samples_cnt)
                writer.add_scalar(f'mAP/task{task_eval + 1}', mAP, samples_cnt)
            
            # Write the average mAP of all tasks to tensorboard
            average_mAP = sum(task_mAP_list) / float(len(task_mAP_list))
            writer.add_scalar("Average mAP", average_mAP, samples_cnt)

    # After training one epoch is done, save the model
    if args.upperbound:
        save_path = os.path.join("model_checkpoints", "joint", "upperbound")
    else:
        save_path = os.path.join("model_checkpoints", "joint", f"seed_{args.seed_num}")
        
    logging.info(f"Saving model at epoch {ep + 1}...")

    # If not exist, create the save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{ep + 1}.pth"))

    # # After training one epoch is done, starts evaluating each task
    # task_eval_results = []
    # for task_eval in [0, 1, 2, 3]:
    #     mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
    #     task_eval_results.append(mAP)
    # epoch_mAP = sum(task_eval_results) / float(len(task_eval_results))
    # epoch_results["epoch_mAP"].append(epoch_mAP)

# Create the save path if not exist
if args.upperbound:
    save_path = os.path.join("outputs", "joint", "upperbound")
else:
    save_path = os.path.join("outputs", "joint", f"seed_{args.seed_num}")

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save the evaluation results
np.save(os.path.join(save_path, "_epoch.npy"), eval_results['epoch'])
np.save(os.path.join(save_path, "_eval.npy"), eval_results['test_mAP'])
np.save(os.path.join(save_path, "_eval_task.npy"), eval_results['task_evaluating'])
np.save(os.path.join(save_path, "_eval_time.npy"), eval_results['data_cnt'])
