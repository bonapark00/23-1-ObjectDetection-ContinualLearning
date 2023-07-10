import logging
import numpy as np
import torch
import torch.nn as nn
from methods.er import ER
from torchvision import transforms
import torch
from tqdm import tqdm
from utils.train_utils import select_pq_dataset, select_stream
from utils.data_loader_clad import SODADataset
from utils.data_loader_shift import SHIFTDataset
from eval_utils.engine import evaluate
import os
import copy

from fast_rcnn.transform import GeneralizedRCNNTransform
import faiss
import time


logger = logging.getLogger()

class RODEO(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        """
            Rodeo method basically offline traines half of the data and online traines the other half.
            In offline training, all the parts of the model is trained, and later detach the front part (front_backbone) of the model.
            Then, with pretrained g_model, we train the pq model. 
            Then in online training, only the back part is used with randomly reconstructed pq features.
        """
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)

        self.pretrain_task_num = kwargs['pretrain_task_num']
        self.codebook_size = kwargs['codebook_size']
        self.memory = select_pq_dataset(self.dataset)
        self.pretrain_task_list = None
        self.g_model = None
        self.pq = None

        clad_task_list = [[0,1,2,3],[2,0,3,1],[1,2,3,0]]
        shift_task_list = [[0,1,2,3,4],[2,0,3,1,4],[4,1,3,0,2]]

        if self.dataset == 'clad':
            assert self.pretrain_task_num <= 4, "pretrain_task_num should be less than 4 in clad"
            selected_tasks = clad_task_list[int(self.seed_num)-1]
            self.pretrain_task_list = selected_tasks[:self.pretrain_task_num]
        
        elif self.dataset == 'shift':
            assert self.pretrain_task_num <= 5, "pretrain_task_num should be less than 5 in shift"
            selected_tasks = shift_task_list[int(self.seed_num)-1]
            self.pretrain_task_list = selected_tasks[:self.pretrain_task_num]

        else:
            raise ValueError("check if the dataset is proper")

    def create_offline_Dataloader(self, dataset, pretrain_task_list, batch_size):
        if dataset == 'clad':
            train_data = SODADataset(path="./dataset/SSLAD-2D", task_ids=pretrain_task_list,
                                        split="train", transforms=transforms.ToTensor(), ssl_required=True)
        else:
            raise ValueError("Shift dataset is not supported yet")
            # TODO: Create Shift offline dataloader.
            # domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy']
            # assert pretrain_task_num <= len(domain_list), "pretrain_task_num should be less than 5 in clad"
            # task_seed_list = [[0,1,2,3,4],[2,0,3,1,4],[4,1,3,0,2]]
            # target_train_tasks = task_seed_list[seed_num-1][:pretrain_task_num]
            # target_domain_dict = {'weather_coarse': [domain_list[i] for i in target_train_tasks]}
            # train_data = SHIFTDataset(domain_dict=target_domain_dict,
            #                             split="train", transforms=transforms.ToTensor())

        dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
        print("Dataloader created")

        return dataloader
    
    def offline_pretrain(self, model, dataloader, optimizer, epochs=1):
        #train the model
        model.train()

        ssl_proposals_list = []
        for epoch in range(epochs):
            for idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} offline training...")):
                images = list(image.to(self.device) for image in images)
                targets_ = targets
                targets = [{'boxes': target['boxes'].to(self.device), 'labels': target['labels'].to(self.device)} for target in targets_]
                ssl_proposals = [{'boxes': target['ssl_proposals'].to(self.device)} for target in targets_]
                ssl_proposals_list.extend([prop['boxes'].cpu() for prop in ssl_proposals])
                
                loss_dict = model(images, targets, ssl_proposals)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if idx % 300 == 0:
                    print(f"Epoch {epoch} Iter {idx} loss: {losses.item()}")

                #remove
                # if idx == 1:
                #     break
                
        print("Offline training is done! successfully!")
        return model, ssl_proposals_list

    def front_backbone_model(self, model):
        backbone = model.backbone
        g_model = nn.Sequential(
             *list(backbone.children())[:-1],
             list(backbone.children())[-1][0])
   
        return g_model
    
    def freeze_front_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    
    def chop_backbone_model(self, model):
        backbone = model.backbone
        chopped_backbone = nn.Sequential(
            *list(backbone.children())[-1][1:])
        model.backbone = chopped_backbone
    

    def extract_backbone_features(self, front_model, dataloader):
        #initialize transform
        transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        front_model.eval()
        images_list = []
        targets_list = []
        front_model_features_list = []

        with torch.no_grad():
            for idx, (images, targets) in enumerate(tqdm(dataloader, desc="Extracting backbone features...")):
                images_list.extend(images)
                images = list(image.to(self.device) for image in images)
                targets_list.extend(targets)
                targets = [{'boxes': target['boxes'].to(self.device), 'labels': target['labels'].to(self.device)} for target in targets]

                images, _ = transform(images, targets)
                features = front_model(images.tensors)
                front_model_features_list.append(features.detach())

                if idx % 100 == 0:
                    print(f"iter {idx} feature extraction is done!")

                #remove
                # if idx == 1:
                #     break

        return images_list, targets_list, front_model_features_list


    def train_pq(self, backbone_features, codebook_size=32, data_dim=2048, nbits=8):
        print(f"training PQ model with codebook size {codebook_size}  nbits {nbits}...")
        assert len(backbone_features) > 0, "backbone_features should be list of features"
    
        base_train_data = []
        for feature in backbone_features:
            detatched_feature = feature.detach().cpu().numpy()
            feature_tr = np.transpose(detatched_feature, (0,2,3,1)).reshape(-1, data_dim).astype('float32')
            base_train_data.append(feature_tr)

        base_train_data = np.concatenate(base_train_data)
        base_train_data = np.ascontiguousarray(base_train_data, dtype='float32')
        print(f"base data is loaded...")

        #train the PQ model
        pq = faiss.ProductQuantizer(data_dim, codebook_size, nbits)

        #remove
        pq.train(base_train_data)
        print(f"PQ model training is done!")

        return pq

    def reconstruct_pq(self, backbone_features, pq_model, data_dim=2048):
        print(f"reconstructing PQ model...")
        assert len(backbone_features) > 0, "backbone_features should be list of features"
        pq = pq_model

        reconstructed_features = []
        for feature in backbone_features:
            _,dim,r,c = feature.shape
            assert dim == data_dim, "data_dim should be same as feature dim"
            detached_feature = feature.detach().cpu().numpy()
            feature_tr = np.transpose(detached_feature, (0,2,3,1)).reshape(-1, data_dim).astype('float32')
            codes = pq.compute_codes(np.ascontiguousarray(feature_tr))
            codes = pq.decode(codes)
            codes = codes.reshape(-1, r, c, data_dim)
            reconstructed_batch = np.transpose(codes, (0,3,1,2))
            assert feature.shape == reconstructed_batch.shape, "dimension does not fit in reconstruction!"
            #split batch to single
            reconstructed_single = np.split(reconstructed_batch, reconstructed_batch.shape[0], axis=0)
            reconstructed_features.extend(reconstructed_single)

        return reconstructed_features


    def online_step(self, sample, sample_num, n_worker):

        #start of training, base initialization
        if sample_num == 1:
            start = time.time()
            print('*'*100)
            print(f"Start offline training of Rodeo method... current dataset: {self.dataset}")
            #dataloader for offline training
            assert self.pretrain_task_list is not None, "pretrain_task_list should be initialized. checkout the dataset."
            offline_dataloader = self.create_offline_Dataloader(self.dataset, self.pretrain_task_list, self.batch_size)
            pretrained_model, ssl_proposals_list = self.offline_pretrain(self.model, offline_dataloader, self.optimizer, epochs=1)
    
            g_model = self.front_backbone_model(self.model)
            images_list, targets_list, front_model_features_list = self.extract_backbone_features(g_model, offline_dataloader)
            pq_model = self.train_pq(front_model_features_list, codebook_size=32, data_dim=2048, nbits=8)   
            reconstructed_features = self.reconstruct_pq(front_model_features_list, pq_model, data_dim=2048)
            g_model = self.freeze_front_model(g_model)      
            
            self.model = pretrained_model
            self.pq = pq_model
            self.g_model = g_model
            self.chop_backbone_model(self.model)

            #add images, targets, ssl_proposals, reconstructed_features to memory
            self.memory.images.extend(images_list)
            self.memory.objects.extend(targets_list)
            self.memory.ssl_proposals.extend(ssl_proposals_list)
            self.memory.pq_features.extend(reconstructed_features)
    
            end = time.time()
            print(f"Offline training is done! It took {end-start} seconds")
            print('*'*100)


        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)


        if sample['task_num'] in self.pretrain_task_list:
            # Only update the memory datalist when the 'task_num' is in the pretrain_task_list
            self.memory.datalist.append(sample)

        else:
            assert self.dataset == 'clad', "Shift dataset is not supported yet"
            img_name = sample['file_name'][:-4]
            ssl_proposals = np.load(os.path.join('precomputed_proposals/ssl_clad', img_name + '.npy'), allow_pickle=True)
            assert ssl_proposals is not None, "Precomputed proposals not found"
            ssl_proposals_tensor = torch.from_numpy(ssl_proposals).to(self.device)

            self.num_updates += self.online_iter
            return_losses, current_pq_feature = self.online_train(sample, ssl_proposals_tensor, self.batch_size, n_worker, iterations=int(self.online_iter))
            self.report_training(sample_num, return_losses, self.writer)
            
            self.update_memory(sample, ssl_proposals_tensor, current_pq_feature)
            self.num_updates -= int(self.num_updates)

    
    def online_train(self, sample, ssl_proposals,  batch_size, n_worker, iterations=1):
        """
            Traines the model using data from the memory. The data is selected randomly from the memory.
        """
       
        total_loss, num_data = 0.0, 0.0
        stream_classname = select_stream(dataset=self.dataset)
        sample_dataset = stream_classname([sample], dataset=self.dataset, transform=None, cls_list=None)
        
        #Note that sample_dataset has only 1 item.
        current_data = sample_dataset.get_data()

        current_image = [current_data['images'][0].to(self.device).type(torch.float)]
        current_target = [{'boxes': current_data['boxes'][0].to(self.device), 'labels': current_data['labels'][0].to(self.device)}]

        #get reconstructed feature from current_image
        transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        current_image_resized, _ = transform(current_image, current_target)
        current_feature = self.g_model(current_image_resized.tensors)

        current_pq_feature = self.reconstruct_pq([current_feature], self.pq, data_dim=2048)
        current_pq_feature_np = current_pq_feature[0]
        current_pq_feature = [torch.from_numpy(feature).to(self.device) for feature in current_pq_feature]
        current_ssl_proposal = [{'boxes': ssl_proposals}]

        if len(self.memory) == 0: 
            raise ValueError("Memory is empty. Please add data to the memory before training the model.")
        
        #current sample should be always contained.
        memory_batch_size = min(batch_size-1, len(self.memory))
        self.count_log += (batch_size)

        self.model.train()
        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size)
            images_memory = [img.to(self.device) for img in memory_data['images']]
            targets_memory = []
            ssl_proposals_memory = []
            
            for i in range(len(images_memory)):
                d = {}
                d['boxes'] = memory_data['boxes'][i].to(self.device)
                d['labels'] = memory_data['labels'][i].to(self.device)
                
                b = {}
                b['boxes'] = memory_data['ssl_proposals'][i].to(self.device)

                targets_memory.append(d)
                ssl_proposals_memory.append(b)
        
            #has to be tensor
            pqs_memory = [pq.to(self.device) for pq in memory_data['pq_features']]

            images_total = current_image + images_memory
            targets_total = current_target + targets_memory
            pq_features_total = current_pq_feature + pqs_memory
            pq_features_batch = torch.cat(pq_features_total, dim=0)
            ssl_proposals_total = current_ssl_proposal + ssl_proposals_memory

            loss_dict = self.model(images_total, targets_total, ssl_proposals=ssl_proposals_total, pq_features=pq_features_batch)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            num_data += len(images_memory)
            self.count_log += memory_batch_size
        
        return (total_loss / iterations) , current_pq_feature_np
        
    def update_memory(self, sample, ssl_proposal, pq_feature):
        # Updates the memory of the model based on the importance of the samples.
        ssl_proposal = ssl_proposal.cpu()
        if len(self.memory.images) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.images))
            self.memory.replace_sample(sample, ssl_proposal, pq_feature, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
            
        else:
            self.memory.replace_sample(sample, ssl_proposal, pq_feature)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
    

    def online_evaluate(self, test_dataloader, sample_num):
        task_list = test_dataloader.dataset.dataset.task_ids
        adjusted_pretrain_list = [idx+1 for idx in self.pretrain_task_list]

        if set(task_list).issubset(set(adjusted_pretrain_list)):
            return 0.0
        
        else:
            eval_backbone = backbone_eval(self.g_model, self.pq, self.model.backbone)
            eval_model = copy.deepcopy(self.model)
            eval_model.backbone = eval_backbone

            coco_evaluator = evaluate(eval_model, test_dataloader, device=self.device)
            stats = coco_evaluator.coco_eval['bbox'].stats
            self.report_test(sample_num, stats[1], self.writer)  # stats[1]: AP @IOU=0.50
            return stats[1]


def collate_fn(batch):
    return tuple(zip(*batch))

class backbone_eval(nn.Module):
    def __init__(self, g_model, pq_model, f_model):
        super().__init__()
        self.g_model = g_model
        self.pq_model = pq_model
        self.f_model = f_model
        self.f_model.eval()

    def reconstruct_pq(self, backbone_feature, pq_model, data_dim=2048):

        pq = pq_model
        reconstructed_feature = []
        feature = backbone_feature
        _,dim,r,c = feature.shape
        assert dim == data_dim, "data_dim should be same as feature dim"
        detached_feature = feature.detach().cpu().numpy()
        feature_tr = np.transpose(detached_feature, (0,2,3,1)).reshape(-1, data_dim).astype('float32')
        codes = pq.compute_codes(np.ascontiguousarray(feature_tr))
        codes = pq.decode(codes)
        codes = codes.reshape(-1, r, c, data_dim)
        reconstructed_batch = np.transpose(codes, (0,3,1,2))
        assert feature.shape == reconstructed_batch.shape, "dimension does not fit in reconstruction!"

        return reconstructed_batch
    
    def forward(self, images):
        x = self.g_model(images)
        x = self.reconstruct_pq(x, self.pq_model)
        x = torch.from_numpy(x).to(images.device)
        x = self.f_model(x)

        return x