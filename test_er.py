import numpy as np
import torch

# Open outputs/npy files
mAP = np.load('./outputs/clad_er/faster_rcnn_clad_bs-16_tbs-8_sd-1_mAP.npy')
eval_task = np.load('./outputs/clad_er/faster_rcnn_clad_bs-16_tbs-8_sd-1_task_evaluating.npy')

# Print last 4 values in each npy
print(mAP[-4:])
print(eval_task[-4:])
breakpoint()
# Get average mAP by averaging four consecutive 4 mAP values
avg_mAP = []
for i in range(0, len(mAP), 4):
    avg_mAP.append(np.mean(mAP[i:i+4]))

# Plot average mAP using matplotlib and save to outputs/upperbound_num_epochs-8_bs16_avg_mAP.png
import matplotlib.pyplot as plt
plt.plot(avg_mAP, label='Average mAP', color='blue')
plt.savefig('./outputs/clad_er/faster_rcnn_clad_bs-16_tbs-8_sd-1_avg_mAP.png')
