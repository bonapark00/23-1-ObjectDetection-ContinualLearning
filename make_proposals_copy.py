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
import selective_search

def load_images_from_subdirectories(path):
    # Get all subdirectories
    subdirectories = [f.path+'/front' for f in os.scandir(path) if f.is_dir()]
    # Initialize an empty list to hold all the images
    all_images = []
    # Iterate over all subdirectories
    # only for the train

    for subdir in subdirectories:
        inside_subdirs = [f.path for f in os.scandir(subdir) if f.is_dir()]
        for subdir2 in inside_subdirs:
            images = glob.glob(os.path.join(subdir2, "*.jpg"))
            all_images.extend(images)
        # Use the glob library to get all the image files in this subdirectory
        # This assumes that the images are .jpg files - change the file extension if needed
        #images = glob.glob(os.path.join(subdir, "*.jpg"))
        # Append these image files to the all_images list
    
    #[0:60200][60200:120400][120400:]
    all_images.sort()
    all_images = all_images[400:]
    #all_images = all_images[0:60200]
    return all_images

def ssl(image_path):
    image = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    rects = rects[0:2000]

    # # Load image as NumPy array from image files
    # image = skimage.io.imread(image_path)
    # # Run selective search using single mode
    # print("generating boxes!")
    # boxes = selective_search.selective_search(image, mode='fast', random_sort=False)
    # boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=2000)
    # print("Done!")
    
    coor_rects = []
    for rect in rects:
        x, y, w, h = rect
        x1, y1, x2, y2 = x, y, x+w, y+h
        rect = [x1, y1, x2, y2]
        coor_rects.append(rect)
    

    img_path = image_path.split('/')
    img_name = '_'.join(img_path[-4:])[:-4]

    np.save('./precomputed_proposals/ssl_shift/'+img_name, coor_rects)


def main():
    #path = './dataset/SHIFT_dataset/discrete/images/train/front/'
    path = './dataset/SHIFT_dataset/discrete/images/'
    os.makedirs('./precomputed_proposals/ssl_shift', exist_ok=True)
    # #visualize boxes and image
    # image = cv2.imread(path)
    # output = image.copy()
    # for (x, y, w, h) in boxes:
    #     # draw the region proposal bounding box on the image
    #     color = [random.randint(0, 255) for j in range(0, 3)]
    #     cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    images = load_images_from_subdirectories(path)
    
    process_num = round(multiprocessing.cpu_count()*0.95)
    with Pool(process_num) as pl:
            start = time.time()
            results = list(tqdm(pl.imap_unordered(ssl, images), total=len(images)))
            end = time.time()
            print("[INFO] selective search took {:.4f} seconds".format(end - start))
            print("[INFO] {} total region proposals".format(sum([r[1] for r in results])))
            print("[INFO] {} output files generated".format(len(results)))

if __name__ == "__main__":
    main()
