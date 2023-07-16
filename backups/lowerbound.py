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

# Transform definition
transform = transforms.ToTensor()

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log', mode='w'), 
                            logging.StreamHandler()])

args = config.finetune_parser()
logging.info(f"Lowerbound training (finetune) with seed {args.seed_num}")

# Set up tensorboard writer
tensorboard_pth = os.path.join(args.tensorboard_pth, "lowerbound", f"seed_{args.seed_num}")
if not os.path.exists(tensorboard_pth):
    os.makedirs(tensorboard_pth)
writer = SummaryWriter(log_dir=tensorboard_pth)

def online_evaluate(model, test_dataloader, sample_num, device):
    coco_evaluator = evaluate(model, test_dataloader, device)
    stats = coco_evaluator.coco_eval['bbox'].stats
    return stats[1]

# Training dataset consists of all tasks, but not shuffled
train_loader_list = []
if not args.debug:
    logging.info("Loading each training dataset...")

    # Load the whole training dataset separately
    for i in range(4):
        dataset = SODADataset(root=args.dataset_root, task_ids=[i+1], split="train")
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                                collate_fn=collate_fn, shuffle=True)
        train_loader_list.append(train_loader)
else:
    # Debug mode
    logging.info("Loading each training debug dataset...")

    # Load the whole training dataset separately
    for i in range(4):
        dataset = SODADataset(root=args.dataset_root, task_ids=[i+1], split="train")
        train_dataset, _ = random_split(dataset, [50, len(dataset) - 50])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                collate_fn=collate_fn, shuffle=True)
        train_loader_list.append(train_loader)

# Always test on all tasks
test_loader_list = []
if not args.debug:
    logging.info("Loading test dataset...")
    for i in range(4):
        dataset = SODADataset(root=args.dataset_root, task_ids=[i+1],
                                    split="val", transforms=transforms.ToTensor())

        test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn))
else:
    logging.info("Loading test debug dataset...")
    for i in range(4):
        dataset = SODADataset(root=args.dataset_root, task_ids=[i+1],
                                    split="val", transforms=transforms.ToTensor())
        debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
        test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=4, collate_fn=collate_fn))

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = select_model("NONE", num_classes=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
model.to(device)

samples_cnt = 0
task_seed_list = [[0,1,2,3], [2,0,3,1],[1,2,3,0]]
selected_seed = task_seed_list[int(args.seed_num) - 1]
eval_results = defaultdict(list)    # Evaluation results (any time inference)
task_record = defaultdict(list)     # Evaluation results after training each task
eval_count = 1

# Train the model, and each task is trained num_iters times to match the number of samples
logging.info("Start training...")
logging.info(f"Finetune, seed_num: {args.seed_num}, num_iters: {args.num_iters}")

for i, task in enumerate(selected_seed):
    # Train one task for num_iters times
    logging.info(f"Start training task {task + 1}...")

    # Train each task for num_iters times
    for it in range(args.num_iters):
        logging.info(f"Task {task + 1}, Iter {it + 1} / {args.num_iters}")
        for data in tqdm(train_loader_list[task], desc=f"Seed {selected_seed}, Task {task + 1}, Iter {it + 1} / {args.num_iters}"):
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
                logging.info(f"LB - Seed {selected_seed}, Task {task + 1}, Iter {it + 1} / {args.num_iters} evaluation")
                task_mAP_list = []
                
                for task_eval in selected_seed[:i + 1]:
                    logging.info(f"While training task {task + 1}, evaluating task {task_eval + 1}...")
                    mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
                    task_mAP_list.append(mAP)
                    eval_results['test_mAP'].append(mAP)
                    eval_results['iter'].append(it + 1)
                    eval_results['task_training'].append(task + 1)
                    eval_results['task_evaluating'].append(task_eval + 1)
                    eval_results['data_cnt'].append(samples_cnt)

                    # Write the evaluation results to the tensorboard   
                    writer.add_scalar(f"mAP/Task {task_eval + 1}", mAP, samples_cnt)
                
                # Writer the average mAP of tasks before and including the current task
                average_mAP = sum(task_mAP_list) / len(task_mAP_list)
                writer.add_scalar(f"Average mAP", average_mAP, samples_cnt)
        
    # When the all iterations of the task are completed, evaluate the model
    logging.info(f"Task {task + 1} training completed, evaluating all tasks...")
    for task_eval in selected_seed:
        mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
        task_record['test_mAP'].append(mAP)
        task_record['task_trained'].append(task + 1)
        task_record['task_evaluating'].append(task_eval + 1)
        task_record['data_cnt'].append(samples_cnt)

# Create path for saving results
eval_save_path = os.path.join('outputs', 'lowerbound', f"seed_{args.seed_num}")
after_save_path = os.path.join('outputs', 'lowerbound', f"after_task", f"seed_{args.seed_num}")
if not os.path.exists(eval_save_path):
    os.makedirs(eval_save_path)
if not os.path.exists(after_save_path):
    os.makedirs(after_save_path)

# Save the evaluation results
np.save(os.path.join(eval_save_path, "mAP.npy"), eval_results['test_mAP'])
np.save(os.path.join(eval_save_path, "iter.npy"), eval_results['iter'])
np.save(os.path.join(eval_save_path, "task_training.npy"), eval_results['task_training'])
np.save(os.path.join(eval_save_path, "task_evaluating.npy"), eval_results['task_evaluating'])
np.save(os.path.join(eval_save_path, "eval_time.npy"), eval_results['data_cnt'])

# Save the evaluation results after training each task
np.save(os.path.join(after_save_path, "mAP.npy"), task_record['test_mAP'])
np.save(os.path.join(after_save_path, "task_trained.npy"), task_record['task_trained'])
np.save(os.path.join(after_save_path, "task_evaluating.npy"), task_record['task_evaluating'])
np.save(os.path.join(after_save_path, "eval_time.npy"), task_record['data_cnt'])
