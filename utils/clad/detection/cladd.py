import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time

from cladd_utils import CladDetection,get_transform
from utils_clad.meta import CLADD_TRAIN_VAL_DOMAINS, CLADD_TEST_DOMAINS, SODA_ROOT
from utils_clad.utils import *
from typing import Optional, List, Sequence, Callable, Dict, Any
from cladd_memory import CladMemoryDataset


def create_match_dict_fn_img(match_dict: Dict[Any, Any]):
    """
    Creates a method that returns true if the image specified by the img_id
    is in the specified domain of the given match_dict.
    
    Args:
        match_dict: dictionary that should match the objects
    
    Return:
        match_fn: a function that evaluates to true if the object is from the given date
    """

    def match_fn(img_id: int, img_dic: Dict[str, Dict]) -> bool:
        img_annot = img_dic[img_id]
        for key, value in match_dict.items():
            if isinstance(value, List):
                if img_annot[key] not in value:
                    return False
            else:
                if img_annot[key] != value:
                    return False
        else:
            return True

    return match_fn


def remove_empty_images(img_ids: list, obj_dict):
    """  
    [Required because torchvision models can't handle empty lists for bbox in targets
    """
    
    non_empty_images = set()
    for obj in obj_dict.values():
        non_empty_images.add(obj["image_id"])
    
    img_ids = [img_id for img_id in img_ids if img_id in non_empty_images]
    return img_ids


def get_matching_detection_info(annot_file: str, match_fn: Callable):
    """
    Creates CLAD task info set according to match_fn
    
    Args:
        annot_file: annotation file from root
        match_fn: A function that takes a sample, the obj and img dicts and return T/F if a sample should be
                     in the dataset
    Return: 
        info set of a single task
    """

    obj_dict, img_dic = load_obj_img_dic(annot_file)
    img_ids = [image for image in img_dic if match_fn(image, img_dic)]
    img_ids = remove_empty_images(img_ids, obj_dict)
    
    return {'img_ids': img_ids, 'annot_file': annot_file}


def get_clad_trainval(root: str=SODA_ROOT, val_proportion= 0.1):
    """
    Selects images which satisfies CLAD domains, and creates info sets of CLAD
    For instance train info set is formed as below
    
        e.g) train info set = [ {'img_ids': [100,124,142], 'annot_file': } ... ]
        
    Args:
        root: root path to the dataset
        val_proportion: proportion of val/train
        
    Return: 
        train info set, val info set
    """
    train_info = []
    val_info = []
    
    #Split from SODA10M, not CLAD-D
    splits = ['train', 'val', 'val', 'val']
    match_fns = [create_match_dict_fn_img(train_domain) for train_domain in CLADD_TRAIN_VAL_DOMAINS]
    
    trainval_info = [get_matching_detection_info( annot_file=os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations',
                                                          f'instance_{split}.json'),
                                                  match_fn = match_fn) for
                    match_fn, split in zip(match_fns, splits)]
    
    
    #split trainval_info to train_info + val_info
    for index,item in enumerate(trainval_info):
        
        all_img_ids = item['img_ids']
        annot_file = item['annot_file']
        task_num = index+1
        cut_off = int((1.0 - val_proportion) * len(all_img_ids))
        
        train_img_ids = all_img_ids[:cut_off]
        val_img_ids = all_img_ids[cut_off: ]
        
        train_info.append({'img_ids': train_img_ids, 'annot_file': annot_file, 'task_num': task_num, 'split': splits[index]})
        val_info.append({'img_ids': val_img_ids, 'annot_file': annot_file, 'task_num': task_num, 'split': splits[index]})
    
    
    return train_info, val_info

        
def get_clad_datalist(data_type: str='train', val_proportion = 0.1):
    
    """
    Creates datalist, so that single data info (sample) can be enumerated as stream.
    All data from CLAD are combined inorder.
    
    Single data is formed as below
    
        e.g) {'file_name': ~.png,  
              'objects': [list of obj annotation info], 
              'task_num': 1, 
              'split': 'train'} 
    Args:
        data_type: 'train' or 'val' (should be extended further for test data)
        val_proportion: proportion of val/train
        
    Return: 
        data list: list that contains single image data.
                   data of all tasks are combined in order. Thus shouldn't be shuffled
    """
    
    datalist = []    
    train_info, val_info = get_clad_trainval(val_proportion=val_proportion)
    selected_info = train_info if data_type == 'train' else val_info
    
    obj_properties = ['image_id','category_id','bbox', 'area', 'id', 'truncated', 'occluded', 'iscrowd']
    empty_obj_properties = dict([(obj_prop, []) for obj_prop in obj_properties])
    
    for item in selected_info:
        img_ids, annot_file, task_num, split = item['img_ids'], item['annot_file'], item['task_num'], item['split']
        obj_container = dict.fromkeys(img_ids)
        obj_dict, img_dict = load_obj_img_dic(annot_file)

        for obj in obj_dict.values():
            if obj['image_id'] in img_ids:
                #the first object from img_ids appears 
                if obj_container[obj['image_id']] == None:
                    file_name = img_dict[obj['image_id']]['file_name']
                    obj_container[obj['image_id']] = \
                        {'file_name': file_name, 
                         'img_info': img_dict[int(obj['image_id'])],
                         'objects': empty_obj_properties,
                         'task_num': task_num, 
                         'split': split }
                    
                    #clear dictionary for later items
                    obj_container[obj['image_id']]['objects']['image_id'] = obj['image_id']
                    empty_obj_properties = dict([(obj_prop, []) for obj_prop in obj_properties])
                    
                for item in obj_properties[1:]:
                    obj_container[obj['image_id']]['objects'][item].append(obj[item])
        
        datalist.extend(obj_container.values())
        
    return datalist
'''
not have test data info yet, also not revised yet

def get_cladd_test(root: str=SODA_ROOT, transform: Callable = None) -> Sequence[CladDetection]:
    """
     Creates the CLAD-D benchmarks train and validation sets, as in the ICCV '21 challenge. This isn't attached to any
     framework and only depends on PyTorch itself. The dataset objects are in COCO format.

    :param root: root path to the dataset
    :param transform: transformation for the test set. If none is given, the default one is used. See
                           `get_tranform`.
    """

    if transform is None:
        transform = get_transform(train=True)

    match_fns = [create_match_dict_fn_img(td) for td in CLADD_TEST_DOMAINS]
    test_sets = [get_matching_detection_set(root,
                                            os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations',
                                                         f'instance_test.json'),
                                            match_fn, transform) for match_fn in match_fns]

    return test_sets
'''

def main():
      start1 = time.time()
      cur_train_datalist = get_clad_datalist(data_type = 'train')
      print(f'{len(cur_train_datalist)} images in total \n')
      end1= time.time()
      
      exposed_classes = [1]
      iteration = 2
      memory = CladMemoryDataset(dataset='SSLAD-2D', device=None)
      
      for i, data in enumerate(cur_train_datalist):
            if not all(item in list(set(data['objects']['category_id']))for item in exposed_classes):
                exposed_classes = list(set(data['objects']['category_id']))
                breakpoint()
                memory.add_new_class(exposed_classes)
                memory.replace_sample(data)
                
                for k in range(iteration): 
                        batch = memory.get_batch(batch_size=1)
                        print(batch)
                        print(memory.datalist)
                        print(memory.images)
                        print(memory.obj_cls_list)
                        print(memory.obj_cls_count)
                        print(memory.obj_cls_train_cnt)
                        print(f'{i} th data, {k} th iter')
                        print()
        
            if i ==3:
                break;
            
      end2 = time.time()
      print(f'{end1-start1} elapsed for data preparation \n')
      
            
if __name__ == "__main__":
    main()