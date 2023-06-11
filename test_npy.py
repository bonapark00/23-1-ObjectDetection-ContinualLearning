import numpy as np
import torch

# Open outputs/npy files
mAP = np.load('./outputs/upperbound_num_epochs-8_bs16_eval.npy')
eval_time = np.load('./outputs/upperbound_num_epochs-8_bs16_eval_time.npy')
eval_task = np.load('./outputs/upperbound_num_epochs-8_bs16eval_task.npy')
epoch_mAP = np.load('./outputs/upperbound_num_epochs-8_bs16_epoch_mAP.npy')

# Get average mAP by averaging four consecutive 4 mAP values
avg_mAP = []
for i in range(0, len(mAP), 4):
    avg_mAP.append(np.mean(mAP[i:i+4]))

# Plot average mAP using matplotlib and save to outputs/upperbound_num_epochs-8_bs16_avg_mAP.png
import matplotlib.pyplot as plt
plt.plot(avg_mAP, label='Average mAP', color='blue')
plt.savefig('./outputs/upperbound_num_epochs-8_bs16_avg_mAP.png')

# Show the mean of 10 last mAP values
print("Mean of 10 last mAP values:", np.mean(mAP[-10:]))

breakpoint()
