import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import select_model
from utils.data_loader_shift import SHIFTMemoryDataset, SHIFTStreamDataset
from utils.visualize import visualize_bbox
from eval_utils.engine import evaluate

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class SHIFT_ER:
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs):
        # Member variables from original er_baseline - ER class
        self.mode = kwargs['mode']
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.exposed_classes = []
        self.seen = 0

        self.dataset = kwargs["dataset"]
        self.device = device

        self.model_name = kwargs["model_name"]
        self.memory_size = kwargs["memory_size"]

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        self.seed_num = kwargs['seed_num']
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        
        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.memory = SHIFTMemoryDataset(dataset='SHIFTDataset', device=None)
        # self.imp_update_period = kwargs['imp_update_period']
        
        self.current_trained_images = []
        self.exposed_tasks = []
        self.count_log = 0
        
        self.model = select_model(mode=self.mode, num_classes=self.n_classes).to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)
        self.task_num = 0
        self.writer = SummaryWriter("tensorboard")
        self.tensorboard_pth = f"{kwargs['mode']}_{self.model_name}_{self.dataset}_b_size{self.batch_size}_tb_size{self.temp_batchsize}_sd_{self.seed_num}"

    
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
            
        self.write_tensorboard(sample)

        # update_memory 호출 -> samplewise_importance_memory 호출 -> 여기에서 memory.replace_sample 호출
        # self.memory.replace_sample(sample)
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if self.num_updates >= 1:
            if len(self.temp_batch) == self.temp_batchsize:
                train_loss = self.online_train(self.temp_batch, self.batch_size, n_worker, 
                                    iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                
                print(f"Train_loss: {train_loss}")
                for sample in self.temp_batch:
                    self.update_memory(sample)

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)
        
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        """Trains the model using both memory data and new data.

        Args:
            sample (_type_): self.temp_batch (samples)
            batch_size (_type_): _description_
            n_worker (_type_): _description_
            iterations (int, optional): _description_. Defaults to 1.
            stream_batch_size (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        total_loss, num_data = 0.0, 0.0
        sample_dataset = SHIFTStreamDataset(sample, dataset="SHIFTDataset", transform=None, cls_list=None)
        memory_batch_size = 0
        
        if len(self.memory) > 0 and batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            images_stream = []; images_memory = []
            targets_stream = []; targets_memory = []

            # Get stream data from SHIFTStreamDataset
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                images_stream = [img.to(self.device) for img in stream_data['images']]
                for i in range(len(images_stream)):
                    d = {}
                    d['boxes'] = stream_data['boxes'][i].to(self.device)
                    d['labels'] = stream_data['labels'][i].to(self.device)
                    targets_stream.append(d)
            
            # Get memory data from SHIFTMemoryDataset
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                images_memory = [img.to(self.device) for img in memory_data['images']]
                for i in range(len(images_memory)):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets_memory.append(d)
            
            # Concat stream data and memory data
            images = images_stream + images_memory
            targets = targets_stream + targets_memory

            # Train
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Report loss
            if self.count_log % 10 == 0:
                task_info = self.train_info()
                logging.info(f"{task_info} - Step {self.count_log}, Current Loss: {losses}")
            self.writer.add_scalar("Loss/train", losses, self.count_log)
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            num_data += len(images)
            self.count_log += (memory_batch_size + stream_batch_size)

            # self.current_trained_images = list(set(self.current_trained_images + memory_data['images']))
            # print("Current trained images:", len(self.current_trained_images))  
            

        return total_loss / iterations
    

    def report_training(self, sample_num, train_loss, writer, log_interval=10):
        writer.add_scalar(f"train/loss", train_loss, sample_num)
        if sample_num % log_interval == 0:
            logger.info(
                f"Train | Sample # {sample_num} | Loss {train_loss:.4f}"
            )

    def report_test(self, sample_num, average_precision):
        writer.add_scalar(f"test/AP", average_precision, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | AP {average_precision:.4f}"
        )

    def online_evaluate(self, test_dataloader, sample_num):
        coco_evaluator = evaluate(self.model, test_dataloader, device=self.device)
        stats = coco_evaluator.coco_eval['bbox'].stats
        self.report_test(sample_num, stats[1])  # stats[1]: AP @IOU=0.50
        return stats[1]
        
    def update_memory(self, sample):
        # Updates the memory of the model based on the importance of the samples.
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)

            # temp_batch update now done in online_step                   
            # if len(self.temp_batch) < self.temp_batchsize:
            #     self.temp_batch.append(target_idx)
                
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
            
            # temp_batch update now done in online_step 
            # if len(self.temp_batch) < self.temp_batchsize:
            #     self.temp_batch.append(len(self.memory)- 1)
        
        
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


    def write_tensorboard(self, sample):
        if sample['task_num'] != self.task_num:
            self.writer.close()     
            self.writer = SummaryWriter(f"tensorboard/{self.tensorboard_pth}")
        
        self.task_num = sample['task_num']
        
    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass

    
    def train_info(self):
        message = f"{self.mode}_{self.dataset}_bs-{self.batch_size}_tbs-{self.temp_batchsize}_sd-{self.seed_num}"
        return message
    
            
def collate_fn(batch):
    return tuple(zip(*batch))
