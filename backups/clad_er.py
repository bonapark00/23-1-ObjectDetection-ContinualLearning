import logging
import torchvision
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.data_loader_clad import CladMemoryDataset

from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLAD_ER:
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        self.memory_size = kwargs["memory_size"]
        self.batch_size = kwargs["batchsize"]
        
        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.n_classes = n_classes
        self.memory = CladMemoryDataset(dataset='SSLAD-2D', device=None)
        self.imp_update_period = kwargs['imp_update_period']
        
        self.current_trained_images = []
        self.exposed_classes = []
        self.exposed_tasks = []
        self.count_log = 0
        
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=n_classes).to(self.device)
        self.params =[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)
        self.task_num = 0
        self.writer = SummaryWriter("tensorboard")
        self.replay_method = kwargs['replay_method'] #base, er
        self.seed_num = kwargs['seed_num']
        self.num_updates = 0
        self.er_num = 2
        self.online_iter = 2
        self.current_batch = []
    
    def online_step(self, sample, sample_num, n_worker):
        """Updates the model based on new data samples. If the sample's class is new, 
        it is added to the model's class set. The memory is updated with the new sample, 
        and the model is trained if the number of updates meets a certain threshold.

        Args:
            sample
            sample_num (int): Sample count for all tasks
            n_worker (int): Number of worker, default zero
        """
        
        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)
            
        # update_memory 호출 -> samplewise_importance_memory 호출 -> 여기에서 memory.replace_sample 호출
        # self.memory.replace_sample(sample)
        self.update_memory(sample)
        self.num_updates += self.online_iter
        
            
        if sample['task_num'] != self.task_num:
            self.writer.close()
        
            num = self.er_num if self.replay_method == 'er' else 0            
            self.writer = SummaryWriter(f"tensorboard/task{self.task_num+1}{self.replay_method}{num}_memory")
        
        self.task_num = sample['task_num']
        
        if self.num_updates >=1:
            if self.replay_method == 'er':
                if len(self.current_batch) == self.er_num:
                    self.num_updates = (self.num_updates//self.er_num)
                    for i in range(self.er_num):
                        train_loss = self.online_train(sample, self.batch_size, n_worker, 
                                            iterations=int(self.num_updates))
                        print(f"Train_loss: {train_loss}")
                      
                    self.num_updates -= int(self.num_updates)
                    self.current_batch.clear()  
            else:
                train_loss = self.online_train(sample, self.batch_size, n_worker, 
                                           iterations=int(self.num_updates))
                print(f"Train_loss: {train_loss}")
                self.num_updates -= int(self.num_updates)
                #self.scheduler.step()
    
    def online_train(self, sample, batch_size, n_worker, iterations=1):
        """Trains the model using both memory data and new data.

        Args:
            sample (_type_): _description_
            batch_size (_type_): _description_
            n_worker (_type_): _description_
            iterations (int, optional): _description_. Defaults to 1.
            stream_batch_size (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if len(self.memory) > 0 and batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size)
        
        for i in range(iterations):
            
            if len(self.memory) > 0 and batch_size > 0:
                
                memory_data = self.memory.get_batch(memory_batch_size, concat_idx=self.current_batch)
                self.count_log += memory_batch_size
                
                self.current_trained_images = list(set(self.current_trained_images + memory_data['images']))
                print("Current trained images:", len(self.current_trained_images))
                
                images = [img.to(self.device) for img in memory_data['images']]
                targets = []
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets.append(d)
         
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                
                #report loss
                if self.count_log % 10 == 0:
                     logging.info(f"Step {self.count_log}, Current Loss: {losses}")
                self.writer.add_scalar("Loss/train", losses, self.count_log)
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                total_loss += losses.item()
                num_data += len(images)
                
        return total_loss / iterations
                
        
    def update_memory(self, sample):
        self.samplewise_importance_memory(sample)
        
        
    def add_new_class(self, class_name):
        """Called when a new class of data is encountered. It extends the model's final layer 
        to account for the new class, and updates the optimizer and memory accordingly.
        
        1. Modify the final layer to handle the new class
        2. Assign the previous weight to the new layer
        3. Modify the optimizer while preserving previous optimizer states

        Args:
            class_name (str): The name of the new class to be added
        """
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        # Updates the loss of the model based on the sample data.
        pass
        
    def samplewise_importance_memory(self, sample):
        # Updates the memory of the model based on the importance of the samples.
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
            
            if self.replay_method == 'er' and len(self.current_batch) < self.er_num:
                self.current_batch.append(target_idx)
                
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
            
            if self.replay_method == 'er' and len(self.current_batch) < self.er_num:
                self.current_batch.append(len(self.memory)- 1)
        
    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass
