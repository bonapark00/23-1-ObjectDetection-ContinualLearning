import numpy as np
import torch
import os

mode = "ilod"
dataset = "clad"
# batchsize = 16
# temp_batchsize = 8
sd = 3

# # Parse each method's outputs/npy files
# # 1. Results during training each task - any time evaluation
# any_time_path = os.path.join('outputs', mode, f"{model_name}_{dataset}_bs-{batchsize}_tbs-{temp_batchsize}_sd-{sd}")
# any_mAP = np.load(any_time_path + "_mAP.npy")
# any_task_training = np.load(any_time_path + "_task_training.npy")
# any_task_evaluating = np.load(any_time_path + "_task_evaluating.npy")
# any_eval_time = np.load(any_time_path + "_eval_time.npy")

# 2. Results after training each task
after_task_path = os.path.join('outputs', mode, f"{dataset}", f"sd-{sd}", "after_task")
after_mAP = np.load(os.path.join(after_task_path, "mAP.npy"))
after_task_trained = np.load(os.path.join(after_task_path, "task_trained.npy"))
after_task_evaluating = np.load(os.path.join(after_task_path, "task_evaluating.npy"))
after_eval_time = np.load(os.path.join(after_task_path, "eval_time.npy"))

# 3. Results for the lowerbound (finetune)
# after_task_path = os.path.join('outputs', "lowerbound", "after_task", f"seed_{sd}/")
# after_mAP = np.load(after_task_path + "mAP.npy")
# after_task_trained = np.load(after_task_path + "task_trained.npy")
# after_task_evaluating = np.load(after_task_path + "task_evaluating.npy")
# after_eval_time = np.load(after_task_path + "eval_time.npy")


# Make after_mAP into 4 x 4 matrix and transpose it, also round it to 2 decimal places
after_mAP = after_mAP.reshape(4, 4).T
after_mAP = np.round(after_mAP, 4)

# Save as csv
# Make row and column names
row_names = ["Task 1", "Task 2", "Task 3", "Task 4"]
col_names = ["Task 1", "Task 2", "Task 3", "Task 4"]
# Save as csv
# np.savetxt(after_task_path + "_mAP.csv", after_mAP, delimiter=",", header=",".join(col_names), comments="", fmt="%s")

# For lowerbound save as csv
np.savetxt(os.path.join(after_task_path, "mAP.csv"), after_mAP, delimiter=",", header=",".join(col_names), comments="", fmt="%s")

# breakpoint()
# # Plot average mAP using matplotlib and save to outputs/upperbound_num_epochs-8_bs16_avg_mAP.png
# import matplotlib.pyplot as plt
# plt.plot(avg_mAP, label='Average mAP', color='blue')
# plt.savefig('./outputs/upperbound_num_epochs-8_bs16_avg_mAP.png')

# # Show the mean of 10 last mAP values
# print("Mean of 10 last mAP values:", np.mean(mAP[-10:]))

# breakpoint()
