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
domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy']
test_loader_list = []
print("Loading test dataset...")
for i in range(len(domain_list)):
    dataset = SHIFTDataset(task_num=i+1, domain_dict={'weather_coarse': domain_list[i]},
                                        split="minival", transforms=transforms.ToTensor())
    test_loader_list.append(torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=collate_fn))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create csv files to write results (taskwise mAP, mean mAP)
eval_results_file = open(f"./upperbound_results.csv", "w")
eval_results_writer = csv.writer(eval_results_file)
eval_results_writer.writerow(["epoch", "task1", "task2", "task3", "task4", "task5", "mean_mAP"])


model = select_model(mode="shift_er", num_classes=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
model.to(device)
# Load the model checkpoint if it exists
for i in range(1, 14):
    print(f"Loading model checkpoint {i}")
    model.load_state_dict(torch.load(f"model_checkpoints/shift/joint/upperbound/epoch_{i}.pth"))

    # Eval
    csv_row = [i]
    for task_eval in range(len(domain_list)):
        print(f"Epoch {i} / {args.num_epochs}, Task {task_eval + 1} Evaluation")
        mAP = online_evaluate(model, test_loader_list[task_eval], i, device)
        csv_row.append(mAP)
        print(f"Task {task_eval + 1} mAP: {mAP}")
    
    mean_mAP = sum(csv_row[1:]) / float(len(csv_row[1:]))
    csv_row.append(mean_mAP)
    eval_results_writer.writerow(csv_row)
    eval_results_file.flush()
    os.fsync(eval_results_file.fileno())


breakpoint()
for ep in range(args.num_epochs):
    print(f"Epoch {ep + 1} / {args.num_epochs}")
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
        #     print(f"Jointly training, upperbound: {args.upperbound}, seed_num: {args.seed_num}, num_epochs: {args.num_epochs}")
        #     print(f"Current Epoch: {ep + 1} / {args.num_epochs}, Step {samples_cnt}, Current Loss: {losses}")

        #     task_mAP_list = []
        #     for task_eval in range(len(domain_list)):
        #         print(f"Epoch {ep + 1} / {args.num_epochs}, Task {task_eval + 1} Evaluation")
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

    print(f"Saving model at epoch {ep + 1}...")

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

# print("Training is done! Saving the evaluation results...")
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
