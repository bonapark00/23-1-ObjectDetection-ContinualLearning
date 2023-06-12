import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torchvision

from models import mnist, cifar, imagenet

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
    if mode == "clad_mir":
        default_config['separate_loss'] = True
        
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes, **default_config)
    return model
