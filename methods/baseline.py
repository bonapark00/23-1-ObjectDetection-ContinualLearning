import logging
import numpy as np
from methods.er import ER

logger = logging.getLogger()

class BASELINE(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        """
            BASELINE class denotes the baseline method where the memory is filled with stream data, and the
            model is trained using data only from the memory. The data from the memory is selected randomly.
        """
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        

    def online_step(self, sample, sample_num, n_worker):
        """Updates the model based on new data samples. Unlike other methods, this method trains the model
           whenever new data is given.

        Args:
            sample
            sample_num (int): Sample count for all tasks
            n_worker (int): Number of worker, default zero
        """
        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)
        
        # We don't need a temp_batch since we train the model whenever new data is given. Also, we update the memory
        # whenever new data is given.
        self.num_updates += self.online_iter
        self.update_memory(sample)
        self.num_updates -= int(self.num_updates)

        # Train the model using data from the memory
        train_loss = self.online_train(self.batch_size, n_worker, iterations=int(self.online_iter))
        self.report_training(sample_num, train_loss, self.writer)
        self.num_updates -= int(self.num_updates)

    
    def online_train(self, batch_size, n_worker, iterations=1):
        """
            Traines the model using data from the memory. The data is selected randomly from the memory.
        """
        total_loss, num_data = 0.0, 0.0

        if len(self.memory) == 0:
            raise ValueError("Memory is empty. Please add data to the memory before training the model.")
        
        memory_batch_size = min(batch_size, len(self.memory))

        # Train the model using data from the memory
        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size)
            images_memory = [img.to(self.device) for img in memory_data['images']]
            targets_memory = []

            for i in range(len(images_memory)):
                d = {}
                d['boxes'] = memory_data['boxes'][i].to(self.device)
                d['labels'] = memory_data['labels'][i].to(self.device)
                targets_memory.append(d)
        
            loss_dict = self.model(images_memory, targets_memory)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            num_data += len(images_memory)
            self.count_log += memory_batch_size
        
        return total_loss / iterations
        
            
def collate_fn(batch):
    return tuple(zip(*batch))
