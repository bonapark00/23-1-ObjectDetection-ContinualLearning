from configuration import config
from utils.preprocess_clad import collate_fn
from utils.data_loader_clad import SODADataset
from utils.train_utils import select_model
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

if not args.is_eval:
    joint_dataset = SODADataset(path="./dataset/SSLAD-2D", task_id=None, split="train")
    joint_dataloader = torch.utils.data.DataLoader(joint_dataset, batch_size=args.batch_size, 
                                                collate_fn=collate_fn, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = select_model("frcnn", None, 7, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003)
    model.to(device)

    count_log = 0

    for ep in range(args.num_epochs):
        for i, data in enumerate(tqdm(joint_dataloader)):
            count_log += len(data[0])
            images = list(transform(img).to(device) for img in data[0])
            targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if count_log % 10 == 0:
                logging.info(f"Step {count_log}, Current Loss: {losses}")
            writer.add_scalar("Loss/train", losses, count_log)

    # Save trained model
    torch.save(model.state_dict(), args.save_pth)


# Evaluation
test_loader_list = []

# TODO: make below code as function
for i in range(4): 
    data_set = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                split="val")

    test_loader_list.append(torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, collate_fn=collate_fn))

print(f"Upperbound model (jointly trained) is on evaluation...")
for i, task in enumerate([0, 1, 2, 3]):
    print(f"Evaluating task {task + 1}...")
    model.load_state_dict(torch.load(args.save_pth))
    model.to(device)
    model.eval()
    
    test_loader = test_loader_list[task]
    evaluate(model, test_loader, device=device)