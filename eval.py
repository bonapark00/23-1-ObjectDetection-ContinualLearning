import torch
import torchvision
import argparse
from PIL import Image
from soda import SODADataset
from clad_utils import get_model_instance_segmentation,collate_fn, visualize_and_save, apply_nms, data_transform
from engine import train_one_epoch, evaluate

test_loader_list = []

for i in range(4): 
        data_set = SODADataset(path="./dataset/SSLAD-2D", task_id=i+1,
                                  split="val", transforms=data_transform)
    
        test_loader_list.append(torch.utils.data.DataLoader(data_set, batch_size=4, collate_fn=collate_fn))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

task = [0,1,2,3]
acum_task = []
task_list = [[0,1,2,3],[2,0,3,1],[1,2,3,0]]
seed_num = [1,2,3]

for i in range(3):
    for item in task_list[i]:
        print("_"*50)
        print(f'model for task {item+1} is on evaluation') 
        print("_"*50)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=7, for_distillation=True)
        model.load_state_dict(torch.load(f"./model_checkpoints/clad_der_model_task_{item+1}_seed{seed_num[i]}.pth"))
        model.to(device)
        model.eval()
        acum_task.append(item)
        
        for k in acum_task:
            test_loader = test_loader_list[k]
            evaluate(model, test_loader, device=device)
    acum_task.clear() 