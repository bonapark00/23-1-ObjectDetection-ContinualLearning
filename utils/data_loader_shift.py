import logging.config
import os
import PIL
import numpy as np
from torchvision import transforms
from utils.data_loader import MemoryDataset
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
import h5py
import torch
from torch.utils.data import DataLoader
from utils.preprocess_shift import get_sample_objects, get_shift_datalist, load_label_img_dic
from torch.utils.data import Dataset



class SHIFTStreamDataset(Dataset):
    """SHIFT dataset for streaming data."""
    def __init__(self, datalist, root, transform, cls_list, device=None, transform_on_gpu=False):
        self.images = []
        self.labels = []
        self.objects=[]
        # self.dataset=dataset
        self.root = root
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = os.path.join(self.root, '/SHIFT_dataset/discrete/images/train/front')
        self.device = device
        self.transform_on_gpu = transform_on_gpu
 
        for data in datalist:
            try:
                img_name=data['file_name']
                
            except KeyError:
                # img_name=data['image_name']
                print("KeyError")
            

            if self.data_dir is not None:
                img_path = data['file_name']
            
            # breakpoint()
            image=PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)
            target = get_sample_objects(data['objects'])
            self.images.append(image)
            self.objects.append(target)

        

    def __len__(self):
        return len(self.images)
    
    @torch.no_grad()
    def get_data(self):
        images = [self.images[idx] for idx in range(len(self.images))]
        boxes = [self.objects[idx]['boxes'] for idx in range(len(self.images))]
        labels = [self.objects[idx]['labels'] for idx in range(len(self.images))]

        return  {'images': images, 'boxes': boxes, 'labels': labels}
    

    
class SHIFTMemoryDataset(MemoryDataset):

    def __init__(self, root, transform=None, cls_list=None, device=None, test_transform=None,
				 transform_on_gpu=False, save_test=None, keep_history=False):
        
        self.datalist = []
        self.images=[]
        self.objects=[]

        self.obj_cls_list=[1]
        self.obj_cls_count = np.zeros(np.max(self.obj_cls_list), dtype=int)
        self.obj_cls_train_cnt = np.zeros(np.max(self.obj_cls_list),dtype=int)
        self.others_loss_decrease = np.array([]) 
        self.previous_idx = np.array([], dtype=int)
        self.device = device
        self.keep_history = keep_history

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.value()
				
        image = self.images[idx]
		
        if self.transform:
            image = self.transform(image)

		
        target = self.objects[idx]
        target = {'boxes':target['boxes'], 'labels':target['labels']}
        return image, target
    
    def show_memory_state(self):
        print("#"*50)
        print("Samples in SHIFTMemoryDataset")
        print("Total Number of Samples: ", len(self.images))
        print("Total Number of Classes: ", len(self.obj_cls_count))
        print("sample image name...")

        for data in self.datalist:
            img_name=data['name']
            print(img_name)
        print("#"*50)

    def add_new_class(self, obj_cls_list):
		
        '''
		when appeard new class, check whether to extend obj_cls_count
		if it is, extend new spaces for new classes('category_id') 
		'''
        self.obj_cls_list = obj_cls_list 
	
        if np.max(obj_cls_list) > len(self.obj_cls_count):
            extend_length = np.max(obj_cls_list)-len(self.obj_cls_count)
            self.obj_cls_count = np.pad(self.obj_cls_count, (0,extend_length), constant_values=(0)).flatten()
            self.obj_cls_train_cnt = np.pad(self.obj_cls_train_cnt, (0,extend_length), constant_values=(0)).flatten()

    def replace_sample(self, sample, idx=None):

        obj_cls_info=np.array(sample['objects']['category_id'])
        obj_cls_id=np.bincount(obj_cls_info)[1:]
        obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
        self.obj_cls_count += obj_cls_id

        try:
            img_name=sample['file_name']
            
        except KeyError:
            img_name=sample['filepath']
        
        img_path = img_name
            

        # if self.data_dir is not None:
        #     img_path = os.path.join("dataset", self.dataset, "discrete", "images", sample['split'], "front", img_name)

        image = PIL.Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image)
        target = get_sample_objects(sample['objects'])
        
        if idx is None:
            self.datalist.append(sample)
            self.images.append(image)
            self.objects.append(target)
        else:

            discard_sample=self.datalist[idx]
            # dis_sample_img=discard_sample['name']
            dis_sample_obj=discard_sample['objects']



            # breakpoint()cont
            obj_cls_info=np.array(dis_sample_obj["category_id"])
            obj_cls_id=np.bincount(obj_cls_info)[1:]
            obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
            self.obj_cls_count -= obj_cls_id


            self.datalist[idx]=sample
            self.images[idx]=image
            self.objects[idx]=target

    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False, transform=None):
        rand_pool=np.array([k for k in range(len(self.images))])
        indices = np.random.choice(rand_pool, size=batch_size, replace=False).astype('int32')
					
		
        images = [self.images[idx] for idx in indices]
        boxes = [self.objects[idx]['boxes'] for idx in indices]		
        labels = [self.objects[idx]['labels'] for idx in indices]
			
		
        for obj in np.array(self.objects)[indices]:   
            obj_cls_id = np.bincount(np.array(obj['labels'].tolist()))[1:]
            if len(self.obj_cls_train_cnt)> len(obj_cls_id):
                obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_train_cnt)-len(obj_cls_id)), constant_values=(0)).flatten()
            self.obj_cls_train_cnt += obj_cls_id

            if self.keep_history:
                self.previous_idx = np.append(self.previous_idx, indices)
                # self.others_loss_decrease = np.append(self.others_loss_decrease, 0)

        return  {'images': images, 'boxes': boxes, 'labels': labels}
    
    def get_two_batches(self, batch_size, test_transform):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False).astype('int32')

        images = [self.images[idx] for idx in indices]
        boxes = [self.objects[idx]['boxes'] for idx in indices]
        labels = [self.objects[idx]['labels'] for idx in indices]
        data_1 = {'images': images, 'boxes': boxes, 'labels': labels}

        images = [self.images[idx] for idx in indices]
        boxes = [self.objects[idx]['boxes'] for idx in indices]
        labels = [self.objects[idx]['labels'] for idx in indices]
        data_2 = {'images': images, 'boxes': boxes, 'labels': labels}

        return data_1, data_2

class SHIFTDistillationMemory(MemoryDataset):

    def __init__(self, root, transform=None, cls_list=None, device=None, test_transform=None,
             data_dir=None, transform_on_gpu=False, save_test=None, keep_history=False):
        
        # breakpoint()
        self.datalist = []
        self.images=[]
        self.objects=[]
        self.proposals=[]
        self.class_logits=[]
        self.box_regression=[]

        self.root = root

        self.obj_cls_list=[1]
        self.obj_cls_count = np.zeros(np.max(self.obj_cls_list), dtype=int)
        self.obj_cls_train_cnt = np.zeros(np.max(self.obj_cls_list),dtype=int)
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.device = device

        self.data_dir =  data_dir
        self.keep_history = keep_history
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
       
        if torch.is_tensor(idx):
           idx = idx.value()
        image=self.images[idx]

        if self.transform:
            image=self.transform(image)
        
        target=self.objects[idx]
        target={'boxes':target['boxes'], 'labels':target['labels']}
        return image, target
    
    def add_new_class(self, obj_cls_list):
        # breakpoint()
        self.obj_cls_list = obj_cls_list

        if np.max(obj_cls_list) > len(self.obj_cls_count):
            extend_length = np.max(obj_cls_list)-len(self.obj_cls_count)
            self.obj_cls_count = np.pad(self.obj_cls_count, (0,extend_length), constant_values=(0)).flatten()
            self.obj_cls_train_cnt = np.pad(self.obj_cls_train_cnt, (0,extend_length), constant_values=(0)).flatten()

    def replace_sample(self, sample, logit, idx=None):
            
            obj_cls_info=np.array(sample['objects']['category_id'])
            obj_cls_id=np.bincount(obj_cls_info)[1:]
            obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
            self.obj_cls_count += obj_cls_id
    
            try:
                img_name=sample['file_name']
                
            except KeyError:
                img_name=sample['filepath']
            
            img_path = img_name
                
    
            # if self.data_dir is not None:
            #     img_path = os.path.join("dataset", self.dataset, "discrete", "images", sample['split'], "front", img_name)
    
            image=PIL.Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)
            target = get_sample_objects(sample['objects'])
            
            if idx is None:
                self.datalist.append(sample)
                self.images.append(image)
                self.objects.append(target)
                self.proposals.append(logit['proposals'])
                self.class_logits.append(logit['class_logits'])
                self.box_regression.append(logit['box_regression'])
            else:
    
                discard_sample=self.datalist[idx]
                # dis_sample_img=discard_sample['name']
                dis_sample_obj=discard_sample['objects']
    
    
    
                # breakpoint()cont
                obj_cls_info=np.array(dis_sample_obj["category_id"])
                obj_cls_id=np.bincount(obj_cls_info)[1:]
                obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
                self.obj_cls_count -= obj_cls_id
    
    
                self.datalist[idx]=sample
                self.images[idx]=image
                self.objects[idx]=target
                self.proposals[idx]=logit['proposals']
                self.class_logits[idx]=logit['class_logits']
                self.box_regression[idx]=logit['box_regression']

    def get_weight(self):
        pass


    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False, transform=None):
        if use_weight:
            print("use_weight is not supported in SHIFTDistillationMemory")
        else:

            indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)

        images=[self.images[idx] for idx in indices]
        boxes=[self.objects[idx]['boxes'] for idx in indices]
        labels=[self.objects[idx]['labels'] for idx in indices]
        proposals=[self.proposals[idx] for idx in indices]
        class_logits=[self.class_logits[idx] for idx in indices]
        box_regression=[self.box_regression[idx] for idx in indices]

        for obj in np.array(self.objects)[indices]:
            obj_cls_id=np.bincount(np.array(obj['labels'].tolist()))[1:]

            if len(self.obj_cls_train_cnt)> len(obj_cls_id):
                obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_train_cnt)-len(obj_cls_id)), constant_values=(0)).flatten()
            
            self.obj_cls_train_cnt += obj_cls_id

        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
            # self.others_loss_decrease = np.append(self.others_loss_decrease, 0)

        return {'images': images, 'boxes': boxes, 'labels': labels, 'proposals': proposals, 'class_logits': class_logits, 'box_regression': box_regression}



class ShiftPQDataset(SHIFTDistillationMemory):
    def __init__(self, root, transform=None, cls_list=None, device=None, test_transform=None,
             data_dir=None, transform_on_gpu=False, save_test=None, keep_history=False,
             pretrain_task_list = None, memory_size = None, total_task_list = None):
        super().__init__(root, transform, cls_list, device, test_transform, data_dir, transform_on_gpu, save_test, keep_history)
        self.datalist = []
        self.images=[]
        self.objects=[]
        self.ssl_proposals = []
        self.pq_features = []
        self.memory_size = memory_size
        self.random_indices = None
        self.total_task_list = total_task_list
        self.total_sample_cnt = 0
        #self.pre_data_idx = [] unlike clad, it is inefficient to track all indexes. calculate_task_idx does this job
        self.shift_train_task = np.array([37041, 25433, 13596, 39016, 25966]) #num of data from task 0
        self.accumulated_train_task = []

        # dataset
        assert bool(pretrain_task_list) * bool(memory_size) != 0, "Pretrain task list and memory size should be given together"
        self.pretrain_task_list = pretrain_task_list
        self.prepare_pretrained_data(self.pretrain_task_list, memory_size=self.memory_size)
        self.pre_create_data_idx(self.total_task_list)

        #rearrange tasklist
        self.shift_train_task = self.shift_train_task[np.array(self.total_task_list)]
        
        num_cnt = 0
        for num in self.shift_train_task:
            self.accumulated_train_task.append(num_cnt)
            num_cnt += num
        

    def prepare_pretrained_data(self, pretrain_task_list, memory_size):
        pass
    
    # call whenever new sample is passed
    def calculate_task_idx(self, data_cnt):
        #index starts with 0.
        task_id = np.digitize(data_cnt, self.accumulated_train_task)
        data_boundary = self.accumulated_train_task[task_id-1]
        idx = data_cnt - data_boundary
        data_cnt +=1 
        
        return {'task_id': task_id, "idx": idx}
           

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
       
        if torch.is_tensor(idx):
           idx = idx.value()
        image=self.images[idx]

        if self.transform:
            image=self.transform(image)
        
        target=self.objects[idx]
        target={'boxes':target['boxes'], 'labels':target['labels']}
        return image, target
    
    def add_new_class(self, obj_cls_list):
        # breakpoint()
        self.obj_cls_list = obj_cls_list

        if np.max(obj_cls_list) > len(self.obj_cls_count):
            extend_length = np.max(obj_cls_list)-len(self.obj_cls_count)
            self.obj_cls_count = np.pad(self.obj_cls_count, (0,extend_length), constant_values=(0)).flatten()
            self.obj_cls_train_cnt = np.pad(self.obj_cls_train_cnt, (0,extend_length), constant_values=(0)).flatten()

    def replace_sample(self, sample, logit, idx=None):
            
            obj_cls_info=np.array(sample['objects']['category_id'])
            obj_cls_id=np.bincount(obj_cls_info)[1:]
            obj_cls_id = np.pad(obj_cls_id, (0,len(self.obj_cls_count)-len(obj_cls_id)), constant_values=(0)).flatten()
            self.obj_cls_count += obj_cls_id
    
            try:
                img_name=sample['file_name']
                
            except KeyError:
                img_name=sample['filepath']
    

class SHIFTDataset(Dataset):

    def __init__(self, root='./dataset',
                 task_num=1, domain_dict={'weather_coarse':'clear'}, split="train", transforms=None, ssl_required=False, pq_required=False):
        self.root=root
        self.split=split
        self.transforms=transforms
        self.ssl_required = ssl_required
        self.task_num = task_num

        self.img_paths=[]
        json_path = os.path.join(self.root, 'SHIFT_dataset', 'discrete', 'images', self.split, 'front', 'det_2d.json')
        self.data_infos_total = load_label_img_dic(json_path)
        # self.data_infos_total = load_label_img_dic(f"{self.root}/{self.split}/front/det_2d.json")
        # Filter out data that does not belong to the domain_dict
        if domain_dict is not None:
            self.data_infos = []
            for data_info in self.data_infos_total:
                if all([data_info['attributes'][key] == domain_dict[key] for key in domain_dict.keys()]):
                    self.data_infos.append(data_info)
        else:
            self.data_infos = self.data_infos_total # If domain_dict is None, use all data

        self.pq_required = pq_required
        self.pq_path = None
        
        if self.pq_required:
            pq_features_path = f'./rodeo_feature/shift_test_{self.task_num}.h5'
            if os.path.exists(pq_features_path):
                data_h5 = h5py.File(pq_features_path, 'r')
                data_num = len(data_h5.keys())
                assert data_num == len(self.img_paths), "PQ features and data num is not matched"
                data_h5.close()

            self.pq_path = pq_features_path

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        boxes, labels=[], []
        img_path = os.path.join(self.root, 'SHIFT_dataset', 'discrete', 'images', self.split, 'front',
                                self.data_infos[idx]['videoName'], self.data_infos[idx]['name'])
        # img_path = f"{self.root}/{self.split}/front/{self.data_infos[idx]['videoName']}/{self.data_infos[idx]['name']}"
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img=self.transforms(img)

        target={}

        # target['img_path'] = img_path # No need to pass img_path
        target['proposal_path'] = f"precomputed_proposals/ssl_shift/{self.split}_front_{self.data_infos[idx]['videoName']}_{self.data_infos[idx]['name'][:-4]}.npy"
        target['boxes'] = torch.as_tensor(self.data_infos[idx]["bboxes"], dtype=torch.float32)
        target['labels'] = torch.tensor(self.data_infos[idx]["labels"], dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = (target['boxes'][:,3]-target['boxes'][:,1])*(target['boxes'][:,2]-target['boxes'][:,0])
        target['iscrowd'] = torch.zeros((len(target['boxes']),), dtype=torch.int64)

        if self.ssl_required:
            ssl_proposals = np.load(target['proposal_path'], allow_pickle=True)
            assert ssl_proposals is not None, "Precomputed proposals not found"
            ssl_proposals = torch.from_numpy(ssl_proposals)
            target["ssl_proposals"] = ssl_proposals
        
        if self.pq_required:
            data_h5 = h5py.File(self.pq_path, 'r')
            pq_features = data_h5[str(idx)][()]
            pq_features = torch.from_numpy(pq_features)
            target['pq_features'] = pq_features
   
        return img, target



def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
        
        #CLAD dataset
        "SSLAD-2D"
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
        
        #not calculated yet. Same as imagenet1000
        "CLAD": (0.485, 0.456, 0.406)
    
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
        
        #not calculated yet. Same as imagenet1000 
        "CLAD": (0.229, 0.224, 0.225),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
        
        #CLAD dataset
        "CLAD": 6,
        #SHIFT
        # "SHIFT": 
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
        
        #CLAD dataset
        "CLAD": 3,
        "SHIFT": 3
        
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
        
        #CLAD dataset. input size differs
        "CLAD": None

    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )