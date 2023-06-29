import torch
import numpy as np
from configuration import config
from torchvision import transforms
from torch.utils.data import random_split
from collections import defaultdict
from tqdm import tqdm
import os
import PIL
import skimage
import cv2
import glob
import time
import random
import multiprocessing
from multiprocessing import Pool

def load_images_from_subdirectories(path):
    # Get all subdirectories
    subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]
    # Initialize an empty list to hold all the images
    all_images = []
    
    # Iterate over all subdirectories
    for subdir in subdirectories:
        # Use the glob library to get all the image files in this subdirectory
        # This assumes that the images are .jpg files - change the file extension if needed
        images = glob.glob(os.path.join(subdir, "*.jpg"))
        
        # Append these image files to the all_images list
        all_images.extend(images)

    return all_images

def selective_search(image_path):
    image = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rects = rects[:128]

    # Create path if it doesn't exist
    os.makedirs('./rects/dataset/SSLAD-2D/labeled/train/', exist_ok=True)
    np.save('./rects/'+ image_path[2:-4], rects)


def main():
    #path = './dataset/SHIFT_dataset/discrete/images/train/front/'
    path = './dataset/SSLAD-2D/labeled/'
    images = load_images_from_subdirectories(path)
 
    process_num = round(multiprocessing.cpu_count()*0.8)
    with Pool(process_num) as p:
            start = time.time()
            results = list(tqdm(p.imap_unordered(selective_search, images), total=len(images)))
            end = time.time()
            print("[INFO] selective search took {:.4f} seconds".format(end - start))
            print("[INFO] {} total region proposals".format(sum([r[1] for r in results])))
            print("[INFO] {} output files generated".format(len(results)))

if __name__ == "__main__":
    main()
