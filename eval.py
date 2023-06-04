import torch
import torchvision
from PIL import Image
from utils.data_loader_clad import SODADataset
from utils.method_manager import select_method
from clad_utils import collate_fn, data_transform
from configuration import config
from engine import evaluate


args = config.base_parser()
test_loader_list = []

# TODO: make below code as function
for i in range(4): 
        data_set = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                  split="val", transforms=data_transform)
    
        test_loader_list.append(torch.utils.data.DataLoader(data_set, batch_size=4, collate_fn=collate_fn))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

task_list = [[0,1,2,3],[2,0,3,1],[1,2,3,0]]
selected_task = task_list[args.seed_num -1]

for idx, item in enumerate(selected_task):
        print("_"*80)
        print(f'model for task {item+1} is on evaluation')
        print("_"*80)
        model = select_method(args, None, device, train_transform, test_transform, 7).model
        model.load_state_dict(torch.load(f"./model_checkpoints/{args.mode}_seed_{args.seed_num}_task{task + 1}.pth"))
        model.to(device)
        model.eval()
        
        #evaluates accumulated task
        for j in selected_task[:idx+1]:
            test_loader = test_loader_list[j]
            evaluate(model, test_loader, device=device)