import numpy as np
import torch

# Open outputs/npy files
array = np.load('outputs/clad_er_faster_rcnn_clad_b_size16_tb_size8_sd_1_eval.npy')
task_eval_array = np.load('outputs/clad_er_faster_rcnn_clad_b_size16_tb_size8_sd_1_eval_task.npy')

# Show the average of all results
print(np.average(array))