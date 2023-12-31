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
from operator import itemgetter
from fast_rcnn.transform import GeneralizedRCNNTransform
import faiss
import time
import h5py

from fast_rcnn.fast_rcnn import fastrcnn_resnet50_fpn

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
        logger.info("RODEO method is used")
        self.pretrain_task_num = kwargs['pretrain_task_num']
        self.codebook_size = kwargs['codebook_size']
        self.pretrain_task_list = None
        self.g_model = None
        self.pq = None
        self.random_indices = None

        clad_task_list = [[0,1,2,3],[2,0,3,1],[1,2,3,0]]
        shift_task_list = [[0,1,2,3,4],[2,0,3,1,4],[4,1,3,0,2]]
        
        #the order will be changed according to the seed
        self.shift_domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy'] 
        
        self.selected_task_list = None
        self.sample_cnt = 0
        self.current_task = 0
        
        if self.dataset == 'clad':
            assert self.pretrain_task_num <= 4, "pretrain_task_num should be less than 4 in clad"
            self.selected_task_list = clad_task_list[int(self.seed_num)-1]
            selected_tasks = [task+1 for task in self.selected_task_list] #Since task_num starts from 1
            self.pretrain_task_list = selected_tasks[:self.pretrain_task_num]


        elif self.dataset == 'shift':
            assert self.pretrain_task_num <= 5, "pretrain_task_num should be less than 5 in shift"
            self.selected_task_list= shift_task_list[int(self.seed_num)-1]
            selected_tasks = [task+1 for task in self.selected_task_list] #Since task_num starts from 1
            self.pretrain_task_list = selected_tasks[:self.pretrain_task_num]
            
            #modify shift domain list according to the selected task list
            modified_domain = np.array(self.shift_domain_list).reshape(-1,1)[self.selected_task_list] 
            self.shift_domain_list = list(modified_domain.reshape(-1))

        else:
            raise ValueError("check if the dataset is proper")
        
        self.memory = select_pq_dataset(self.memory_size, self.pretrain_task_list, self.selected_task_list, self.dataset, self.root)
        
        
    def create_offline_Dataloader(self, dataset, pretrain_task_list, batch_size, split='train'):
        if dataset == 'clad':
            train_data = SODADataset(root=self.root, task_ids=pretrain_task_list,
                                        split=split, transforms=transforms.ToTensor(), ssl_required=True)
            
        elif dataset == 'shift':
            initial_domain_list = ['clear', 'cloudy', 'overcast', 'rainy', 'foggy'] 
            if split == 'train':
                print("Rodeo only assumes that shift dataset is splitted into weather coarse. if not please modify the code")
                data_list = []
                for idx, domain in zip(pretrain_task_list, self.shift_domain_list) :
                    single_dataset = SHIFTDataset(root=self.root, task_num=idx, domain_dict={'weather_coarse': domain},
                                                split="train", transforms=transforms.ToTensor(), ssl_required=True)
                    data_list.append(single_dataset)
                    
                train_data = ConcatDataset(*data_list)

            elif split == 'minival':
                train_data = SHIFTDataset(root=self.root, task_num=pretrain_task_list[0], domain_dict={'weather_coarse': initial_domain_list[pretrain_task_list[0]-1]},
                                          split="minival", transforms=transforms.ToTensor(), ssl_required=True)
            else:
                raise ValueError("check if the dataset is proper")
        else:
            raise ValueError("check if the dataset is proper")
            
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
        print("Dataloader created")
        return dataloader
    
    
    def offline_pretrain(self, model, dataloader, optimizer, epochs=16):
        #train the model
        model.train()
        for epoch in range(epochs):
            for idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} offline training...")):
                
                images = [image.to(self.device) for image in images]
                targets_modified = [{'boxes': target['boxes'], 'labels': target['labels']} for target in targets]
                ssl_proposals = [target['ssl_proposals'] for target in targets]
                
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets_modified]
                ssl_proposals = [{'boxes': prop.to(self.device)} for prop in ssl_proposals]
    
                loss_dict = model(images, targets, ssl_proposals)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if idx % 100 == 0:
                    logger.info(f"Epoch {epoch} Iter {idx} loss: {losses.item()}")
                    
                # if idx == 1:
                #     break
                    
        print("Offline training is done! successfully!")
        return model


    def front_backbone_model(self, model):
        with torch.no_grad():
            backbone = copy.deepcopy(model.backbone)
            g_model = nn.Sequential(
                *list(backbone.children())[:-1],
                list(backbone.children())[-1][0])
   
        return g_model
    
    
    def freeze_front_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
        return model


    def chop_backbone_model(self, model):
        backbone = copy.deepcopy(model.backbone)
        chopped_backbone = nn.Sequential(
            *list(backbone.children())[-1][1:])
        model.backbone = chopped_backbone
    
    
    def extract_backbone_features(self, front_model, dataloader, save_path):
        #initialize transform
        transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        front_model.eval()

        os.makedirs('./rodeo_feature', exist_ok=True)
        if os.path.exists(save_path):
            print("backbone feature file already exists, move onto the next step")
        else: 
            h5_file = h5py.File(save_path, 'w')
            with torch.no_grad():      
                for idx, (images, _) in enumerate(tqdm(dataloader, desc="Extracting backbone features...")):
                    images = [image.to(self.device) for image in images]
                    images, _ = transform(images, None)
                    features = front_model(images.tensors)
                    features = features.cpu().numpy()
                    h5_file.create_dataset(str(idx),data=features)
                    
                    if idx % 100 == 0:
                        print(f"iter {idx} feature extraction is done!")
                        
            h5_file.close()
     
            
    
    def reconstruct_pq_features(self, pq_model, front_model, dataloader, task_index, save_path, data_dim = 2048):

        clad_train_task = [4470, 1329, 1479, 524]
        shift_train_task = [37041, 25433, 13596, 39016, 25966]
        
        pq = pq_model
        transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        front_model.eval()
        
        target_type = 'train' if task_index is None else 'test'
        if target_type == 'train':
            target_task_list = clad_train_task if self.dataset == 'clad' else shift_train_task
        else:
            target_task_list = None
        
        feature_cnt = 0
        current_task = 1
        h5_file = h5py.File(save_path, 'w')
        with torch.no_grad():      
            for idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Extracting pq features of {target_type} {task_index}...")):
                images = [image.to(self.device) for image in images]
                images, _ = transform(images, None)
                features = front_model(images.tensors)
                features = features.cpu().numpy()

                reconstructed_batch = self.reconstruct_direct(features, pq, data_dim=data_dim)
                reconstructed_container = np.split(reconstructed_batch, reconstructed_batch.shape[0], axis=0)
                
                for single_feature in reconstructed_container:
                    if target_task_list is not None:
                        if target_task_list[current_task-1] == feature_cnt:
                            current_task +=1
                            feature_cnt = 0
                    
                        h5_file.create_dataset(f'{str(current_task)} {str(feature_cnt)}',data=single_feature)
                    else:
                        h5_file.create_dataset(str(feature_cnt),data=single_feature)
                        
                    feature_cnt +=1
        h5_file.close()


    def train_pq(self, codebook_size=32, data_dim=2048, nbits=8):
        print(f"training PQ model with codebook size {codebook_size}  nbits {nbits}...")
    
        feature_path = f'./rodeo_feature/{self.dataset}_{self.pretrain_task_list}_backbone_train.h5'
        data_h5 = h5py.File(feature_path, 'r')
        keys = list(data_h5.keys())
        
        base_train_data = []
        for batch_id in tqdm(keys):
            feature = data_h5[batch_id][()]
            feature_tr = np.transpose(feature, (0,2,3,1)).reshape(-1, data_dim).astype('float32')
            base_train_data.append(feature_tr)
        data_h5.close()

        base_train_data = np.concatenate(base_train_data)
        base_train_data = np.ascontiguousarray(base_train_data, dtype='float32')
        print(f"base data is loaded...")
   
        #train the PQ model
        pq = faiss.ProductQuantizer(data_dim, codebook_size, nbits)
        start = time.time()
        #remove
        pq.train(base_train_data)
        end = time.time()
        print(f"PQ model training is done! It took {end-start} seconds")
        
        #erase backbone extracted file, since it is not needed anymore
        os.remove(feature_path)
        return pq
    
    def reconstruct_direct(self, backbone_feature, pq_model, data_dim=2048):

        pq = pq_model
        feature = backbone_feature
        _,dim,r,c = feature.shape
        assert dim == data_dim, "data_dim should be same as feature dim"
        feature_tr = np.transpose(feature, (0,2,3,1)).reshape(-1, data_dim).astype('float32')
        codes = pq.compute_codes(np.ascontiguousarray(feature_tr))
        codes = pq.decode(codes)
        codes = codes.reshape(-1, r, c, data_dim)
        reconstructed_batch = np.transpose(codes, (0,3,1,2))
        assert feature.shape == reconstructed_batch.shape, "dimension does not fit in reconstruction!"

        return reconstructed_batch
        

    def reconstruct_all_pq_features(self, pq_model, front_model, train_dataloader, test_dataloader_list, data_dim=2048):
        print(f"reconstructing all features of data... it will take a while")
    
        #train dataset
        #train dataloader should never be shuffled. Index computed in order.
        assert train_dataloader is not list, "train_dataloader should be a total singledataloader"
        train_feature_path = f'./rodeo_feature/{self.dataset}_reconstructed_train_features.h5'
        self.reconstruct_pq_features(pq_model, front_model, train_dataloader, None, train_feature_path)
        
        #test dataset
        assert isinstance(test_dataloader_list, list), "test_dataloader should be prepared in list, with ordered task index"
        for idx, test_dataloader in enumerate(test_dataloader_list):
            test_feature_path = f'./rodeo_feature/{self.dataset}_test_{idx+1}.h5'
            self.reconstruct_pq_features(pq_model, front_model, test_dataloader, idx+1, test_feature_path)
        
        print("all features reconstruction is done!")
        

    def online_step(self, sample, sample_num, n_worker):

        #start of training, base initialization
        if sample_num == 1:
            start = time.time()
            print('*'*100)
            print(f"Start offline training of Rodeo method... current dataset: {self.dataset}")
            
            #dataloader for offline training
            assert self.pretrain_task_list is not None, "pretrain_task_list should be initialized. checkout the dataset."
            offline_dataloader = self.create_offline_Dataloader(self.dataset, self.pretrain_task_list, self.batch_size)
            pretrained_model = self.offline_pretrain(self.model, offline_dataloader, self.optimizer, epochs=16) #self.batch_size
            g_model = self.front_backbone_model(self.model)
            
            # #extract backbone features
            backbone_path = f'./rodeo_feature/{self.dataset}_{self.pretrain_task_list}_backbone_train.h5'
            self.extract_backbone_features(g_model, offline_dataloader, backbone_path)
            
            # #train pq model
            pq_model = self.train_pq(codebook_size=self.codebook_size, data_dim=2048, nbits=8)   
            
            # #reconstruct all features and save as h5 file
            # #prepare dataloaders
            total_task_num = 4 if self.dataset == 'clad' else 5
            train_dataloader = self.create_offline_Dataloader(self.dataset,[i+1 for i in range(total_task_num)], self.batch_size, split='train')
            test_dataloader_list = []
            for i in range(total_task_num):
                split_category = 'minival' if self.dataset == 'shift' else 'test'
                test_dataloader = self.create_offline_Dataloader(self.dataset,[i+1], self.batch_size, split=split_category)
                test_dataloader_list.append(test_dataloader)
                
            self.reconstruct_all_pq_features(pq_model, g_model, train_dataloader, test_dataloader_list)
            
            
            #prepare for online training
            g_model = self.freeze_front_model(g_model)      
            self.model = pretrained_model
            self.pq = pq_model
            self.g_model = g_model
            self.chop_backbone_model(self.model)
            
            #reinitialize optimizer 
            self.params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(self.params, lr=0.0001, weight_decay=0.0003)
            logger.info(f"optimizer initialized")
            
            end = time.time()
            print(f"Offline training is done! It took {end-start} seconds")
            print('*'*100)


        if not set(sample['objects']['category_id']).issubset(set(self.exposed_classes)):
            self.exposed_classes = list(set(self.exposed_classes + sample['objects']['category_id']))
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(self.exposed_classes)


        if int(sample['task_num']) not in self.pretrain_task_list:
            img_name = sample['file_name'][:-4]
            if self.dataset == 'clad': 
                ssl_proposals = np.load(os.path.join('precomputed_proposals/ssl_clad', img_name + '.npy'), allow_pickle=True)
            
            elif self.dataset == 'shift':
                parsed_img_name = img_name.split('/')[-4:]
                joined_img_name = '_'.join(parsed_img_name) + '.npy'
                ssl_proposals = np.load(os.path.join('precomputed_proposals/ssl_shift', joined_img_name), allow_pickle=True)
        
            assert ssl_proposals is not None, "Precomputed proposals not found"
            ssl_proposals_tensor = torch.from_numpy(ssl_proposals).to(self.device)

            self.num_updates += self.online_iter
            if self.current_task != sample['task_num']:
                self.current_task = sample['task_num']
                self.sample_cnt = 0
            
            train_feature_path = f'./rodeo_feature/{self.dataset}_reconstructed_train_features.h5'
            losses = self.online_train(sample, ssl_proposals_tensor, self.batch_size, train_feature_path, n_worker, iterations=int(self.online_iter))
            self.report_training(sample_num, losses, self.writer)
            self.update_memory(sample)
            self.num_updates -= int(self.num_updates)
            self.sample_cnt +=1
            
    
    def online_train(self, sample, ssl_proposals, batch_size, train_feature_path, n_worker, iterations=1):
        """
            Traines the model using data from the memory. The data is selected randomly from the memory.
        """
        h5_file = h5py.File(train_feature_path, 'r')
        total_loss, num_data = 0.0, 0.0
        stream_classname = select_stream(dataset=self.dataset)
        sample_dataset = stream_classname([sample], root=self.root, transform=None, cls_list=None)
        
        #Note that sample_dataset has only 1 item.
        #prepare all the data for training
        current_data = sample_dataset.get_data()
        current_image = [current_data['images'][0].to(self.device).type(torch.float)]
        current_target = [{'boxes': current_data['boxes'][0].to(self.device), 'labels': current_data['labels'][0].to(self.device)}]

        #get reconstructed feature from current_image        
        current_feature_id = f'{str(self.current_task)} {str(self.sample_cnt)}'
        current_pq_feature = [torch.from_numpy(h5_file[current_feature_id][()]).to(self.device)]
        current_ssl_proposal = [{'boxes': ssl_proposals}]

        if len(self.memory) == 0: 
            raise ValueError("Memory is empty. Please add data to the memory before training the model.")
        
        #current sample should be always contained.
        memory_batch_size = min(batch_size-1, len(self.memory))
        self.count_log += (batch_size)

        self.model.train()
        for i in range(iterations):
            memory_data = self.memory.get_batch(memory_batch_size, train_feature_path)
            images_memory = [img.to(self.device) for img in memory_data['images']] #images are still needed for resizing targets 
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
        
        return (total_loss / iterations)
        
        
    def update_memory(self, sample):
   
        if len(self.memory.datalist) >= self.memory_size:
            target_idx = np.random.randint(len(self.memory.datalist))
            self.memory.replace_sample(sample, target_idx)
            self.dropped_idx.append(target_idx)
            self.memory_dropped_idx.append(target_idx)
            
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory)- 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)
    

    def online_evaluate(self, test_dataloader, sample_num):
        task_list = test_dataloader.dataset.task_ids
        adjusted_pretrain_list = self.pretrain_task_list

        if set(task_list).issubset(set(adjusted_pretrain_list)):
            return 0.0
        else:         
            eval_model = copy.deepcopy(self.model)
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

    def reconstruct_direct(self, backbone_feature, pq_model, data_dim=2048):
        pq = pq_model
        reconstructed_feature = []
        feature = backbone_feature
        _,dim,r,c = feature.shape
        assert dim == data_dim, "data_dim should be same as feature dim"
        feature_tr = np.transpose(feature, (0,2,3,1)).reshape(-1, data_dim).astype('float32')
        codes = pq.compute_codes(np.ascontiguousarray(feature_tr))
        codes = pq.decode(codes)
        codes = codes.reshape(-1, r, c, data_dim)
        reconstructed_batch = np.transpose(codes, (0,3,1,2))
        assert feature.shape == reconstructed_batch.shape, "dimension does not fit in reconstruction!"

        return reconstructed_batch
    
    def forward(self, images):
        x = self.g_model(images)
        x = self.reconstruct_direct(x, self.pq_model)
        x = torch.from_numpy(x).to(images.device)
        x = self.f_model(x)

        return x


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.accumulated_size = [0] + np.cumsum([len(d) for d in self.datasets]).tolist()
          
    def __getitem__(self, i):
        dataset_idx = np.digitize(i, self.accumulated_size) - 1
        image_idx = i - self.accumulated_size[dataset_idx]
        image, target = self.datasets[dataset_idx].__getitem__(image_idx)
    
        return image, target

    def __len__(self):
        return sum(len(d) for d in self.datasets)