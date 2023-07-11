import logging
import torchvision
import os 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from methods.er import ER   
from utils.data_loader_clad import CladDistillationMemory, CladStreamDataset, CladMemoryDataset
import PIL
from torchvision import transforms
import torch
from utils.train_utils import select_stream, select_distillation, select_model

logger = logging.getLogger()

class ILOD(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, writer, **kwargs)
        logging.info("ILOD Initialized")
        
        # ILOD Model
        self.model_teacher = None
        self.task_changed = 0
        
        distillation_classname = select_distillation(self.dataset)
        self.memory = distillation_classname(dataset=self.dataset)
        self.temp_ssl_batch = []

    def online_step(self, sample, sample_num, n_worker):
        """Updates the model based on new data samples. If the sample's class is new, 
        it is added to the model's class set. The memory is updated with the new sample, 
        and the model is trained if the number of updates meets a certain threshold.

        Args:
            sample
            sample_num (int): Sample count for all tasks
            n_worker (int): Number of worker, default zero
        """

        # Gives information about seen images / total passed images
        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)
            
        # load precomputed proposals
        img_name = sample['file_name'][:-4]
        ssl_proposals = np.load(os.path.join('precomputed_proposals/ssl_clad', img_name + '.npy'), allow_pickle=True)
        assert ssl_proposals is not None, "Precomputed proposals not found"
        ssl_proposals = torch.from_numpy(ssl_proposals).to(self.device)

        # Need to change as function
        if sample['task_num'] != self.task_num:
            self.task_changed += 1

            # Switch teacher as task changed
            if self.task_changed > 1:
                print('teacher changed!')
                self.model_teacher = select_model(mode=self.mode, num_classes=self.n_classes).to(self.device)
                self.model_teacher.load_state_dict(self.model.state_dict()) # Copy weights from student to teacher
                self.model_teacher.eval()
                self.model_teacher.roi_heads.generate_soft_proposals = True

        self.task_num = sample['task_num']   
        self.temp_batch.append(sample)
        self.temp_ssl_batch.append(ssl_proposals)
        self.num_updates += self.online_iter
        
        if len(self.temp_batch) == self.temp_batchsize:
            # print(self.temp_batchsize)
            # Make ready for direct training (doesn't go into memory before training)
            train_loss = self.online_train(self.temp_batch, self.temp_ssl_batch, self.batch_size, n_worker, 
                                           iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize
                                            )
            self.report_training(sample_num, train_loss, self.writer)

            for sample, proposals in zip(self.temp_batch, self.temp_ssl_batch):
                #here tuple (sample, ssl_proposals) are jointly added to memory
                self.update_memory(sample, proposals)

            self.temp_batch = []
            self.temp_ssl_batch = []
            self.num_updates -= int(self.num_updates)
             

    def online_train(self, sample, ssl_proposals, batch_size, n_worker, iterations=1, stream_batch_size=1):
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
        total_loss, num_data = 0.0, 0.0
        stream_classname = select_stream(dataset=self.dataset)
        sample_dataset = stream_classname(sample, dataset=self.dataset, transform=None, cls_list=None)
        
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        self.count_log += (stream_batch_size + memory_batch_size)
        
        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size) if memory_batch_size > 0 else None
            images_stream = []; images_memory = []
            targets_stream = []; targets_memory = []
            ssl_proposals_stream = []; ssl_proposals_memory = []
            
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                images_stream = [img.to(self.device) for img in stream_data['images']]
                for i in range(len(images_stream)):
                    d = {}
                    d['boxes'] = stream_data['boxes'][i].to(self.device)
                    d['labels'] = stream_data['labels'][i].to(self.device)
                    targets_stream.append(d)

                ssl_proposals_stream = [{'boxes':prop.to(self.device)} for prop in ssl_proposals] if isinstance(ssl_proposals[0], torch.Tensor) else ssl_proposals

            if memory_batch_size > 0:
                #concat data from memory
                images_memory = [img.to(self.device) for img in memory_data['images']]
                for i in range(len(memory_data['images'])):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets_memory.append(d)
                
                ssl_proposals_memory = [{'boxes':prop.to(self.device)} for prop in memory_data['proposals']] #proposals from memory
                # Concat stream data and memory data
                images = images_stream + images_memory
                targets = targets_stream + targets_memory
                ssl_proposals_all = ssl_proposals_stream + ssl_proposals_memory

                # Calculate distillation loss
                if self.model_teacher:
                    with torch.no_grad():
                        _ = self.model_teacher(images, ssl_proposals=ssl_proposals_all)
                    pl_te = self.model_teacher.proposals_logits

                    losses = self.model(images, targets, ssl_proposals_all, pl_te['proposals'])
                    pl_st = self.model.proposals_logits

                    l2_loss = torch.nn.MSELoss()

                    #distillation loss
                    distill_cls = 0
                    for (output, target) in zip(pl_st['class_logits'], pl_te['class_logits']):
                        output = torch.sub(output, torch.mean(output, dim=1).reshape(-1,1)) #subtract mean
                        target = torch.sub(target, torch.mean(target, dim=1).reshape(-1,1)) #subtract mean
                        distill_cls += l2_loss(output, target)

                    distill_reg = 0
                    for (output, target) in zip(pl_st['box_regression'], pl_te['box_regression']):
                        distill_reg += l2_loss(output, target)

                    #hard coded distillation loss
                    distill_loss = 1/(64*7) * (distill_cls.detach() + distill_reg.detach())
                    loss = sum(loss for loss in losses.values()) + distill_loss

                # While first task (do not have any teacher model)
                else:
                    losses = self.model(images, targets, ssl_proposals_all)
                    loss = sum(loss for loss in losses.values())
            else:
                losses = self.model(images_stream, targets_stream, ssl_proposals_stream)
                loss = sum(loss for loss in losses.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return (total_loss / iterations)
 
    def update_memory(self, sample, proposals):
        # Updates the memory of the model based on the importance of the samples.
        proposals = {'proposals': proposals, 'class_logits': None, 'box_regression': None}
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, proposals, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
    
        else:
            self.memory.replace_sample(sample, proposals)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)