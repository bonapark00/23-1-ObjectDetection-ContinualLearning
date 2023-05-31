import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torchvision
import torch
import random
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Optional, List, Sequence, Callable, Dict, Any



class CladDetection(torch.utils.data.Dataset):
    """
    A class that creates a Clad-D dataset, which will covers a given domain. Class incremental style datasets
    isn't supported in this dataset.

    :param root: root path of the folder where files are stored
    :param ids: Ids of the images that should be in the dataset
    :param transform: Transform to be applied to images before returning
    :param meta: Any string with usefull meta information.
    """

    def __init__(self, root: str,
                 ids: Sequence[int],
                 annot_file: str,
                 transform: Optional[Callable] = None,
                 meta: str = None,
                 ):
        super(CladDetection).__init__()

        split = annot_file.split('_')[-1].split('.')[0]

        self.img_folder = os.path.join(root, 'SSLAD-2D', 'labeled', split)
        self.ids = ids
        self.transform = None #transform if transform is not None else get_transform(split == 'train')
        self.meta = meta

        self.obj_annotations, self.img_annotations = load_obj_img_dic(annot_file)
        self._remove_empty_images()
        self.img_anns = self._create_index()

    def _remove_empty_images(self):
        """
        Required because torchvision models can't handle empty lists for bbox in targets
        """
        non_empty_images = set()
        for obj in self.obj_annotations.values():
            non_empty_images.add(obj["image_id"])
        self.ids = [img_id for img_id in self.ids if img_id in non_empty_images]

    def _create_index(self):
        img_anns = defaultdict(list)
        for ann in self.obj_annotations.values():
            img_anns[ann['image_id']].append(ann)
        return img_anns

    def _load_target(self, index: str):
        img_id = self.ids[index]
        img_objects = self.img_anns[img_id]

        boxes = []
        for obj in img_objects:
            bbox = obj["bbox"]
            # Convert from x, y, h, w to x0, y0, x1, y1
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([obj["category_id"] for obj in img_objects], dtype=torch.int64)
        area = torch.as_tensor([obj["area"] for obj in img_objects])
        iscrowd = torch.as_tensor([obj["iscrowd"] for obj in img_objects], dtype=torch.int64)

        # Targets should all be tensors
        target = {"boxes": boxes, "labels": labels, "image_id": torch.as_tensor(img_id, dtype=torch.int64),
                  "area": area, "iscrowd": iscrowd}

        return target

    def _load_image(self, index):
        file_name = self.img_annotations[self.ids[index]]['file_name']
        return Image.open(os.path.join(self.img_folder, file_name)).convert('RGB')

    def __getitem__(self, index):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)





# Below adapted from pytorch vision example on detection, but removed unnecessary code.

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


def get_transform(train):
    transform_arr = [ToTensor()]
    if train:
        transform_arr.append(RandomHorizontalFlip(0.5))
    return Compose(transform_arr)
