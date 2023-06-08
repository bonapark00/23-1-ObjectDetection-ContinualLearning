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

args = config.joint_parser()

# Set up logging
writer = SummaryWriter(args.tensorboard_pth)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log', mode='w'), 
                            logging.StreamHandler()])
save_path = "model_checkpoints"
os.makedirs(save_path, exist_ok=True)

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

test_loader_list = []

if not args.debug:
    for i in range(4): 
        dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                    split="val", transforms=transforms.ToTensor())

        test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn))
else:
    for i in range(4):
        dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                    split="val", transforms=transforms.ToTensor())
        debug_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
        test_loader_list.append(torch.utils.data.DataLoader(debug_dataset, batch_size=4, collate_fn=collate_fn))


if not args.debug:
    joint_dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=None, split="train")
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batch_size, 
                                                collate_fn=collate_fn, shuffle=True)
else:
    debug_joint_dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=None, split="train")
    joint_dataset, _ = random_split(debug_joint_dataset, [70, len(debug_joint_dataset) - 70])
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

for ep in range(args.num_epochs):
    logging.info(f"Epoch {ep + 1} / {args.num_epochs}")
    for i, data in enumerate(tqdm(joint_dataloader)):
        model.train()
        samples_cnt += args.batch_size
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
            logging.info(f"Current Epoch: {ep + 1} / {args.num_epochs}")
            logging.info(f"Step {samples_cnt}, Current Loss: {losses}")
            for task_eval in [0, 1, 2, 3]:
                logging.info(f"Epoch {ep + 1} / {args.num_epochs}, Task {task_eval + 1} Evaluation")
                mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
                eval_results["test_mAP"].append(mAP)
                eval_results["task_eval"].append(task_eval + 1)
                eval_results["data_cnt"].append(samples_cnt)
                writer.add_scalar(f'mAP/task{task_eval + 1}', mAP, samples_cnt)

    # After training one epoch is done, starts evaluating each task
    task_eval_results = []
    for task_eval in [0, 1, 2, 3]:
        mAP = online_evaluate(model, test_loader_list[task_eval], samples_cnt, device)
        task_eval_results.append(mAP)
    epoch_mAP = sum(task_eval_results) / float(len(task_eval_results))
    epoch_results["epoch_mAP"].append(epoch_mAP)

# Save results
save_path = (
    f"upperbound_num_epochs-{args.num_epochs}"
    f"_bs{args.batch_size}"
)
np.save(os.path.join('outputs', save_path + "_eval.npy"), eval_results['test_mAP'])
np.save(os.path.join('outputs', save_path + "eval_task.npy"), eval_results['task_eval'])
np.save(os.path.join('outputs', save_path + "_eval_time.npy"), eval_results['data_cnt'])
np.save(os.path.join('outputs', save_path + "_epoch_mAP.npy"), epoch_results['epoch_mAP'])
# # Save trained model
# torch.save(model.state_dict(), args.save_pth)
