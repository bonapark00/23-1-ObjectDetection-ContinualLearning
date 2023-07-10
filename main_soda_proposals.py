from utils.data_loader_clad import SODADataset
from tqdm import tqdm
import torch
import torchvision
from fast_rcnn.fast_rcnn import fastrcnn_resnet50_fpn
from utils.preprocess_clad import collate_fn, visualize_bbox_ssls
from torchvision import transforms
from eval_utils.engine import evaluate
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

def online_evaluate(model, test_dataloader, sample_num, device):
    coco_evaluator = evaluate(model, test_dataloader, device)
    stats = coco_evaluator.coco_eval['bbox'].stats
    return stats[1]

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

parser = argparse.ArgumentParser()
parser.add_argument("--tensorboard_pth", type=str, default="./tensorboard")
parser.add_argument("--model_pth", type=str, default="./checkpoints/ssl")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

# Set up tensorboard writer
tensorboard_pth = os.path.join(args.tensorboard_pth, "voc_test_ssl" if not args.debug else "voc_test_debug")
if not os.path.exists(tensorboard_pth):
    os.makedirs(tensorboard_pth)
writer = SummaryWriter(log_dir=tensorboard_pth)

# Load dataset
total_dataset = SODADataset(task_ids=[1, 2, 3, 4], split="train", transforms=train_transform, ssl_required=True)

# Split dataset if debug
if args.debug:
    total_dataset, _ = torch.utils.data.random_split(total_dataset, [50, len(total_dataset) - 50])

# Split dataset into train and test
train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [int(0.9 * len(total_dataset)), len(total_dataset) - int(0.9 * len(total_dataset))])
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=args.num_workers)
dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=args.num_workers)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21).to(device)
model = fastrcnn_resnet50_fpn(num_classes=7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)

samples_cnt = 0
for epoch in range(args.num_epochs):
    print("Epoch:", epoch)
    model.train()
    for i, data in enumerate(tqdm(dataloader_train)):
        samples_cnt += len(data[0])
        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
        ssl_proposals = [{k: v.to(device) for k, v in t.items()} for t in data[2]]
        
        losses = model(images, targets, ssl_proposals)
        loss = sum(loss for loss in losses.values())
        
        writer.add_scalar("train/loss", loss.item(), samples_cnt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test every epoch
    model.eval()
    mAP = online_evaluate(model, dataloader_test, samples_cnt, device=device)
    writer.add_scalar("test/mAP", mAP, samples_cnt)
    print("mAP:", mAP)