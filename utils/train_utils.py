import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torchvision
from models import mnist, cifar, imagenet
from fast_rcnn.fast_rcnn import fastrcnn_resnet50_fpn, fastrcnn_resnet50
from utils.data_loader_clad import CladMemoryDataset, CladStreamDataset, CladDistillationMemory, CladPQDataset
from utils.data_loader_shift import SHIFTMemoryDataset, SHIFTStreamDataset, SHIFTDistillationMemory

default_config = {
    'rpn_pre_nms_top_n_train': 2000,
    'rpn_pre_nms_top_n_test': 1000,
    'rpn_post_nms_top_n_train': 2000,
    'rpn_post_nms_top_n_test': 1000,
    'separate_loss': False,
    'generate_soft_proposals': False,
    'soft_num': 64
}

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

#method to select optimizer. 
#model.fc.parameters are added with different param_group. so there are two items in param_group
def select_optimizer(opt_name, lr, model):
    if opt_name == "adam":
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.Adam(params, lr=lr, weight_decay=0)
        opt.add_param_group({'params': model.fc.parameters()})

    elif opt_name == "radam":
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
        opt.add_param_group({'params': model.fc.parameters()})
    elif opt_name == "sgd":
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
        opt.add_param_group({'params': model.fc.parameters()})
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_memory(dataset="clad"):
    if dataset == "clad":
        return CladMemoryDataset
    elif dataset == "shift":
        return SHIFTMemoryDataset
    else:
        raise NotImplementedError("Please select the dataset [clad, shift]")

def select_stream(dataset="clad"):
    if dataset == "clad":
        return CladStreamDataset
    elif dataset == "shift":
        return SHIFTStreamDataset
    else:
        raise NotImplementedError("Please select the dataset [clad, shift]")

def select_distillation(dataset="clad"):
    if dataset == "clad":
        return CladDistillationMemory
    elif dataset == "shift":
        return SHIFTDistillationMemory
    else:
        raise NotImplementedError("Please select the dataset [clad, shift]")
    
def select_pq_dataset(memory_size, pretrain_task_list, total_task_list, dataset="clad", root='./dataset'):
    if dataset == "clad":
        return CladPQDataset(root = root, memory_size = memory_size, pretrain_task_list = pretrain_task_list, total_task_list = total_task_list)

    elif dataset == "shift":
        raise NotImplementedError("shift is not prepared yet")


#method to select learning rate schedulaer.
def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler

# Method to initiliaize the model. model_class is dependend to dataset
#e.g) cifer, imagenet -> Resnet, mnist -> MLP
def select_model(mode="clad_er", num_classes=7):
    if mode == "mir":
        default_config['separate_loss'] = True

    elif mode == 'ilod':
        model = fastrcnn_resnet50_fpn(num_classes=num_classes)
        return model
    
    elif mode == 'rodeo':
        model = fastrcnn_resnet50(num_classes=num_classes)
        return model


    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes, **default_config)
    return model
