import logging
import torchvision
import os 
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from methods.clad_er import CLAD_ER
from utils.data_loader_clad import CladDistillationMemory
import PIL
from torchvision import transforms

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLAD_DER(CLAD_ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory_size = 150 #kwargs["memory_size"]
        self.batch_size = 4
        
        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.n_classes = n_classes
        self.memory = CladDistillationMemory(dataset='SSLAD-2D', device=None)

        # Exposed classes
        self.current_trained_images = []
        self.exposed_classes = []
        self.exposed_tasks = []
        
        #Customized torchvision model. for normal model use for_distillation = False (default)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=n_classes, for_distillation=True).to(self.device)
        self.params =[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)

        self.seed_num = kwargs['seed_num']  #only used for tensorboard
        self.current_batch = []             #batch for stream
    
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
            
        write_tensorboard(sample)
       
        self.current_batch.append(sample)
        self.num_updates += self.online_iter
        
        if len(self.current_batch) == self.temp_batchsize:
            #make ready for direct training (doesn't go into memory before training)
            current_batch_data= [(self.get_sample_img_tar(item)) for item in self.current_batch]

            train_loss, logits= self.online_train(current_batch_data, self.batch_size, n_worker, 
                                                iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize
                                                ,alpha=0.05, beta=0.5, theta=1
                                                )
            print(f"Train_loss: {train_loss}")
            for idx, stored_sample in enumerate(self.current_batch):
                self.update_memory(stored_sample,\
                                   {'proposals': logits['proposals'][idx], 
                                    'class_logits': logits['class_logits'][idx], 
                                    'box_regression': logits['box_regression'][idx]})    

            self.current_batch.clear() 
            self.num_updates -= int(self.num_updates)
             
              
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=2, alpha=0.05, beta=0.5, theta= 1):
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
 
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        self.count_log += stream_batch_size + memory_batch_size

        

        for i in range(iterations):
            self.model.train()
            memory_data = self.memory.get_batch(memory_batch_size) if memory_batch_size >0 else None

            if memory_data:
                self.current_trained_images = list(set(self.current_trained_images + memory_data['images']))
                print("Current trained images:", len(self.current_trained_images), ", not included stream batch")

            images = [s[0] for s in sample]
            targets = [s[1] for s in sample] 

            if memory_batch_size > 0:
                
                #concat data from memory
                images += [img.to(self.device) for img in memory_data['images']]
                for i in range(len(memory_data['images'])):
                    d = {}
                    d['boxes'] = memory_data['boxes'][i].to(self.device)
                    d['labels'] = memory_data['labels'][i].to(self.device)
                    targets.append(d)

                #caution with different length with images, targets
                proposals = [prop.to(self.device) for prop in memory_data['proposals']] #(512,4) tensor
                class_logits = [cl.to(self.device) for cl in memory_data['class_logits']] #(512, 7)
                box_regression = [br.to(self.device) for br in memory_data['box_regression']] #(512, 28)

                losses, proposals_logits, z_logits = self.model(images, targets, proposals)
                
                ##################################################################################
                # z_logits, proposals_logits = {proposals:[(512,4)], class_logits:[(512,7)], ...}
                # proposals_logits: newly updated infos of all data
                # z_logits: logits given from previous proposals (used in distillation loss)
                ##################################################################################

                distill_func = torch.nn.MSELoss()
                #distillation loss
                distill_cls = 0
                for (output, target) in zip(z_logits['class_logits'], class_logits):
                    distill_cls += distill_func(output, target)

                distill_reg = 0
                for (output, target) in zip(z_logits['box_regression'], box_regression):
                    distill_reg += distill_func(output, target)

                distill_loss = (alpha * distill_cls.detach() + beta * distill_reg.detach())/memory_batch_size
                print(f'distill_cl: {(alpha * distill_cls/memory_batch_size)}   distill_rg :{beta * distill_reg.detach()/memory_batch_size}')
                #breakpoint()

                losses, proposals_logits, _ = self.model(images, targets)
                losses['loss_classifier'] = torch.mean(losses['loss_classifier'])
                losses['loss_box_reg'] = torch.mean(losses['loss_box_reg'][0])

                ''' (수정필요함)
                losses['loss_classifier'] = torch.mean(losses['loss_classifier'][:-memory_batch_size]) + theta*torch.mean(losses['loss_classifier'][-memory_batch_size:])
                losses['loss_box_reg'] = torch.mean(losses['loss_box_reg'][:-memory_batch_size][0]) + theta*torch.mean(losses['loss_box_reg'][-memory_batch_size:][0])
                losses['loss_objectness'] = (1+theta)*losses['loss_objectness']
                losses['loss_rpn_box_reg'] = (1+theta)*losses['loss_rpn_box_reg']
                 '''

                loss = sum(loss for loss in losses.values()) + theta*distill_loss
                print(f"CL:{sum(loss for loss in losses.values())}, DL:{distill_loss}\n")

            else:
                losses, proposals_logits, _ = self.model(images, targets)
                losses['loss_classifier'] = torch.mean(losses['loss_classifier'])
                losses['loss_box_reg'] = torch.mean(losses['loss_box_reg'][0])
                
                loss = sum(loss for loss in losses.values())

            #report loss
            if self.count_log % 10 == 0:
                    logging.info(f"Step {self.count_log}, Current Loss: {loss}")
            self.writer.add_scalar("Loss/train", loss, self.count_log)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            for item in ['proposals', 'class_logits', 'box_regression']:
                proposals_logits[item] = proposals_logits[item][:stream_batch_size] 

            return (total_loss/iterations), proposals_logits
        
    def update_memory(self, sample, logit):
        self.samplewise_importance_memory(sample, logit)
        
        
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
        

    def samplewise_importance_memory(self, sample, logit):
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
        
    def get_sample_img_tar(self, sample):
            '''
            return image and target for sample which enumerated from main 
            '''

            #prepare Image
            img_name = sample['file_name']
            img_path = os.path.join("dataset","SSLAD-2D",'labeled',sample['split'],img_name)
            image = PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image).to(self.device)

            #prepare targets
            boxes = []
            for bbox in sample['objects']['bbox']:
                  # Convert from x, y, h, w to x0, y0, x1, y1
                  boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                  
            # Targets should all be tensors
            target = \
                  {"boxes": torch.as_tensor(boxes, dtype=torch.float32).to(self.device), 
                   "labels": torch.as_tensor(sample['objects']['category_id'],dtype=torch.int64).to(self.device)}
            
            
            return image, target


    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        # Adjusts the learning rate of the optimizer based on the learning history.
        pass
    