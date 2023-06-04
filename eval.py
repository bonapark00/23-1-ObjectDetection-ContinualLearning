import torch
import torchvision
from torchvision import transforms
from PIL import Image
from utils.data_loader_clad import SODADataset
from utils.method_manager import select_method
from clad_utils import collate_fn, data_transform
from configuration import config
from engine import evaluate


args = config.base_parser()
test_loader_list = []

# Transform definition
train_transform = transforms.Compose([
transforms.ToTensor()
])

test_transform = transforms.Compose([
transforms.ToTensor()
])

# TODO: make below code as function
for i in range(4): 
    data_set = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                split="val", transforms=data_transform)

    test_loader_list.append(torch.utils.data.DataLoader(data_set, batch_size=4, collate_fn=collate_fn))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

task_seed_list = [[2,0,3,1],[0,1,2,3],[1,2,3,0]]
selected_seed = task_seed_list[args.seed_num - 1]

for i, task in enumerate(task_seed_list[int(args.seed_num) - 1]):
    print(f"Model for task {task + 1} is on evaluation...")
    model = select_method(args, None, device, train_transform, test_transform, 7).model
    model.load_state_dict(torch.load(f"./model_checkpoints/{args.mode}_seed_{args.seed_num}_task{task + 1}.pth"))
    model.to(device)
    model.eval()

    # Evaluates accumulated tasks
    for prev_task in selected_seed[:i + 1]:
        test_loader = test_loader_list[prev_task]
        evaluate(model, test_loader, device=device)

