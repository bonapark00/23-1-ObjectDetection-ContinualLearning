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

#visualize and save bounding boxes with image
def visualize(image_path):
    image = cv2.imread(image_path)
    rects_path = './rects/rects/'+ image_path[2:-4] + '.npy'
    rects = np.load(rects_path)
    output = image.copy()



    for (x, y, w, h) in rects:
		# draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    #save image
    cv2.imwrite('test.png',output)

def main():
    #path = './dataset/SHIFT_dataset/discrete/images/train/front/'
    path = './dataset/SSLAD-2D/labeled/train/HT_TRAIN_000482_SH_000.jpg'
    
    visualize(path)

if __name__ == "__main__":
    main()
