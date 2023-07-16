import logging
import numpy as np
import torch
from methods.er import ER
from torchvision import transforms
from utils.train_utils import select_stream, select_distillation

logger = logging.getLogger()

class DER(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, writer, **kwargs)
        logger.info("DER method is used")
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.theta = kwargs['theta']

        distillation_classname = select_distillation(self.dataset)
        self.memory = distillation_classname(root=self.root)
    
        # Customized torchvision model. for normal model use for_distillation = False (default)
        # TODO: select_model must be called only in the CLAD_ER class
        

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
        
        # Add sample to memory - sample consists of image, objects, and image_id
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        
        if len(self.temp_batch) == self.temp_batchsize:
            # Make ready for direct training (doesn't go into memory before training)

            train_loss, logits = self.online_train(self.temp_batch, self.batch_size, n_worker, 
                                                iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)

            self.report_training(sample_num, train_loss, self.writer)
            for idx, stored_sample in enumerate(self.temp_batch):
                self.update_memory(stored_sample,\
                                   {'proposals': logits['proposals'][idx], 
                                    'class_logits': logits['class_logits'][idx], 
                                    'box_regression': logits['box_regression'][idx]})
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
            
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
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
        sample_dataset = stream_classname(sample, root=self.root, transform=None, cls_list=None)
        
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        self.count_log += (stream_batch_size + memory_batch_size)

        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size) if memory_batch_size > 0 else None
            images_stream = []; images_memory = []
            targets_stream = []; targets_memory = []
            
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                images_stream = [img.to(self.device) for img in stream_data['images']]
                for i in range(len(images_stream)):
                    d = {}
                    d['boxes'] = stream_data['boxes'][i].to(self.device)
                    d['labels'] = stream_data['labels'][i].to(self.device)
                    targets_stream.append(d)
            # breakpoint()
            if memory_batch_size > 0:
                #concat data from memory
                images_memory = [img.to(self.device) for img in memory_data['images']]
                for i in range(len(memory_data['images'])):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets_memory.append(d)

                # Concat stream data and memory data
                images = images_stream + images_memory
                targets = targets_stream + targets_memory

                #caution with different length with images, targets
                proposals = [prop.to(self.device) for prop in memory_data['proposals']] #(512,4) tensor
                class_logits = [cl.to(self.device) for cl in memory_data['class_logits']] #(512, 7)
                box_regression = [br.to(self.device) for br in memory_data['box_regression']] #(512, 28)

                losses = self.model(images, targets, proposals)
                st_logits = self.model.student_logits
                proposals_logits = self.model.proposals_logits
                ###################################################################################
                # st_logits, proposals_logits = {proposals:[(512,4)], class_logits:[(512,7)], ...}#
                # proposals_logits: newly updated infos of all data                               #
                # st_logits: logits given from previous proposals (used in distillation loss)     #
                ################################################################################### 
                l2_loss = torch.nn.MSELoss()
                #distillation loss
                distill_cls = 0
                for (output, target) in zip(st_logits['class_logits'], class_logits):
                    distill_cls += l2_loss(output, target)

                distill_reg = 0
                for (output, target) in zip(st_logits['box_regression'], box_regression):
                    distill_reg += l2_loss(output, target)

                distill_loss = (self.alpha * distill_cls.detach() + self.beta * distill_reg.detach())/memory_batch_size
                loss = sum(loss for loss in losses.values()) + self.theta * distill_loss

            else:
                losses = self.model(images_stream, targets_stream)

                proposals_logits = self.model.proposals_logits
                loss = sum(loss for loss in losses.values())
                
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
      
        for item in ['proposals', 'class_logits', 'box_regression']:
                proposals_logits[item] = proposals_logits[item][:stream_batch_size] 
        

        return (total_loss / iterations), proposals_logits
        
    def update_memory(self, sample, logit):
        # Updates the memory of the model based on the importance of the samples.
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, logit, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
            
        else:
            self.memory.replace_sample(sample, logit)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
    