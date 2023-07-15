import logging
import copy
import torchvision 
import numpy as np
import torch
from methods.er import ER
from utils.train_utils import select_stream, select_model

logger = logging.getLogger()

class MIR(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        logger.info("MIR method is used")
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.cand_size = kwargs['mir_cands']
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        """
        
        """
        total_loss, num_data = 0.0, 0.0

        assert stream_batch_size > 0
        stream_classname = select_stream(dataset=self.dataset)
        sample_dataset = stream_classname(sample, dataset=self.dataset, transform=None, cls_list=None)
        
        for i in range(iterations):
            stream_data = sample_dataset.get_data()
            stream_images = [img.to(self.device) for img in stream_data['images']]
            stream_targets = []
            for i in range(len(stream_images)):
                d = {}
                d['boxes'] = stream_data['boxes'][i].to(self.device)
                d['labels'] = stream_data['labels'][i].to(self.device)
                stream_targets.append(d)

            loss_dict = self.model(stream_images, stream_targets)
            # Calculate total loss
            losses = torch.tensor(0, dtype=torch.float32, device='cuda:0')
            for key, value in loss_dict.items():
                # if the value is a list, convert it to a tensor
                if isinstance(value, list):
                    value = torch.stack(value)

                losses += value.sum()
            self.optimizer.zero_grad()
            losses.backward()

            # Getting the gradients after a forward and backward pass
            grads = {}

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grads[name] = param.grad.data

            if len(self.memory) > 0:
                memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
                lr = self.optimizer.param_groups[0]['lr']
                # new_model = copy.deepcopy(self.model)
                new_model = select_model(mode=self.mode, num_classes=self.n_classes).to(self.device)
                new_model.load_state_dict(self.model.state_dict())
                for name, param in new_model.named_parameters():
                    if param.requires_grad:
                        param.data = param.data - lr * grads[name]

                memory_cands, memory_cands_test = self.memory.get_two_batches(min(self.cand_size, len(self.memory)), test_transform=None)
                images = [img.to(self.device) for img in memory_cands_test['images']]
                targets = []
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = memory_cands_test['boxes'][i].to(self.device)
                    d['labels'] = memory_cands_test['labels'][i].to(self.device)
                    targets.append(d)
                
                # scores = torch.zeros(len(images))
                with torch.no_grad():
                    pre_loss_dict = self.model(images, targets)
                    post_loss_dict = new_model(images, targets)
                    pre_loss_classifier = pre_loss_dict['loss_classifier']
                    pre_loss_box_reg = torch.stack(pre_loss_dict['loss_box_reg'])
                    pre_loss_objectness = torch.stack(pre_loss_dict['loss_objectness'])
                    pre_loss_rpn_box_reg = torch.stack(pre_loss_dict['loss_rpn_box_reg'])

                    post_loss_classifier = post_loss_dict['loss_classifier']
                    post_loss_box_reg = torch.stack(post_loss_dict['loss_box_reg'])
                    post_loss_objectness = torch.stack(post_loss_dict['loss_objectness'])
                    post_loss_rpn_box_reg = torch.stack(post_loss_dict['loss_rpn_box_reg'])

                    samplewise_pre_loss = pre_loss_classifier + pre_loss_box_reg + pre_loss_objectness + pre_loss_rpn_box_reg
                    samplewise_post_loss = post_loss_classifier + post_loss_box_reg + post_loss_objectness + post_loss_rpn_box_reg
                    
                    scores = samplewise_post_loss - samplewise_pre_loss

                
                # Fetches selected samples from the memory_cands, which are the training candidates, 
                # based on the selected sample indices obtained using the test candidates.
                images = [img.to(self.device) for img in memory_cands['images']]
                targets = []
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = memory_cands['boxes'][i].to(self.device)
                    d['labels'] = memory_cands['labels'][i].to(self.device)
                    targets.append(d)

                selected_samples = torch.argsort(scores, descending=True)[:memory_batch_size]
                selected_images = [images[idx.item()] for idx in selected_samples]
                selected_targets = [targets[idx.item()] for idx in selected_samples]

                final_images = stream_images + selected_images
                final_targets = stream_targets + selected_targets

                self.optimizer.zero_grad()
                loss_dict = self.model(final_images, final_targets)
                
                losses = torch.tensor(0, dtype=torch.float32, device='cuda:0')
                for key, value in loss_dict.items():
                    # if the value is a list, convert it to a tensor
                    if isinstance(value, list):
                        value = torch.stack(value)

                    losses += value.sum()

                losses.backward()
                self.optimizer.step()

                total_loss += losses.item()
                num_data += len(images)
                self.count_log += (memory_batch_size + stream_batch_size)
        
        return total_loss / iterations
    
    def update_memory(self, sample):
        # Updates the memory of the model based on the importance of the samples.
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
            
            if len(self.temp_batch) < self.temp_batchsize:
                self.temp_batch.append(target_idx)
                
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
            
            if len(self.temp_batch) < self.temp_batchsize:
                self.temp_batch.append(len(self.memory) - 1)
        
        
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
        
        
    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass
            
def collate_fn(batch):
    return tuple(zip(*batch))
