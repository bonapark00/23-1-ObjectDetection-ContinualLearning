import os
import json
from functools import lru_cache
from itertools import product
from typing import Sequence, Dict
import requests
import zipfile
import torchvision.transforms as T
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from configuration.clad_meta import SODA_DOMAINS
from configuration.clad_meta import CLADD_TRAIN_VAL_DOMAINS, SODA_ROOT
from utils.preprocess_clad import *
from typing import List, Callable, Dict, Any
import torch

# def get_sample_objects(labels:dict):

#     #no need in shift







    

