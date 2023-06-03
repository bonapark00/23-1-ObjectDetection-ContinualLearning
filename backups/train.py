import torch
from torch.utils.data import DataLoader
from utils.soda import SODADataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
from tqdm import tqdm
import logging
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from clad_utils import collate_fn, data_transform, get_model_instance_segmentation

batch_size = 4
dataset_task1_train = SODADataset(path="../SSLAD-2D", task_id=1,
                                  split="train", transforms=data_transform)
dataset_task1_val = SODADataset(path="../SSLAD-2D", task_id=1,
                                  split="val", transforms=data_transform)

train_dataloader = torch.utils.data.DataLoader(dataset_task1_train, batch_size=batch_size, collate_fn=collate_fn, 
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset_task1_val, batch_size=batch_size, collate_fn=collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log', mode='w'), 
                              logging.StreamHandler()])


# Training
model = get_model_instance_segmentation(14)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
save_path = "model_checkpoints"
os.makedirs(save_path, exist_ok=True)

count_log = 0
num_epochs = 5
for ep in range(num_epochs):
    epoch_loss = 0
    model.train()
    for imgs, annotations in tqdm(train_dataloader, desc=f"Epoch {ep+1}/{num_epochs}"):
        count_log += batch_size
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations) 
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        
        if count_log % 10 == 0:
            logging.info(f"Epoch {ep+1}, Step {count_log}, Current Loss: {losses}")

        writer.add_scalar("Loss/train", losses, count_log)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    evaluate(model, test_dataloader, device=device)
    torch.save(model.state_dict(), os.path.join(save_path, f"model_{ep+1}ep.pth"))
    epoch_loss /= len(train_dataloader)
    logging.info(f"Epoch {ep+1} of {num_epochs}, Epoch Loss: {epoch_loss}")
