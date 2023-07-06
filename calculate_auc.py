import numpy as np
import torch
import os
import matplotlib.pyplot as plt

mode = "clad_filod"
model_name = "faster_rcnn"
dataset = "clad"
batchsize = 16
temp_batchsize = 8
seed = [1, 2, 3]
num_tasks = 4 # clad

def find_end_index(arr, current_ptr):
    # Fine the index of the last element that stops increasing
    end_index = current_ptr
    while end_index < len(arr) - 1 and arr[end_index + 1] > arr[end_index]: 
        end_index += 1
    return end_index

for sd in seed:
    # Parse each method's outputs/npy files
    # 1. Results during training each task - any time evaluation
    any_time_path = os.path.join('outputs', mode, f"{model_name}_{dataset}_bs-{batchsize}_tbs-{temp_batchsize}_sd-{sd}")
    any_mAP = np.load(any_time_path + "_mAP.npy")
    any_task_training = np.load(any_time_path + "_task_training.npy")
    any_task_evaluating = np.load(any_time_path + "_task_evaluating.npy")
    any_eval_time = np.load(any_time_path + "_eval_time.npy")

    # Iterate through any_mAP and any_eval_time to get mAP_list and samples_count_list
    mAP_list = []
    samples_count_list = []
    current_ptr = 0
    while current_ptr < len(any_mAP):
        end_index = find_end_index(any_task_evaluating, current_ptr)
        mAP_average = np.mean(any_mAP[current_ptr:end_index+1])
        mAP_list.append(mAP_average)
        samples_count_list.append(any_eval_time[current_ptr])
        current_ptr = end_index + 1

    mAP_AUC = np.mean(mAP_list)
    mAP_last_by_task = any_mAP[-num_tasks:].tolist()
    mAP_last = any_mAP[-1]

    result_dict = {
        'mAP_AUC':mAP_AUC,
        'mAP_last':mAP_last,
        'mAP_last_by_task':mAP_last_by_task,
        'mAP_list':mAP_list,
        'samples_count':samples_count_list,
    }

    # Print mAP_AUC
    print("mAP_AUC:", mAP_AUC)

breakpoint()


# 3. Results for the lowerbound (finetune)
# after_task_path = os.path.join('outputs', "lowerbound", "after_task", f"seed_{sd}/")
# after_mAP = np.load(after_task_path + "mAP.npy")
# after_task_trained = np.load(after_task_path + "task_trained.npy")
# after_task_evaluating = np.load(after_task_path + "task_evaluating.npy")
# after_eval_time = np.load(after_task_path + "eval_time.npy")


# Make after_mAP into 4 x 4 matrix and transpose it, also round it to 2 decimal places
after_mAP = after_mAP.reshape(5, 5).T
after_mAP = np.round(after_mAP, 4)

# Save as csv
# Make row and column names
row_names = ["Task 1", "Task 2", "Task 3", "Task 4", "task 5"]
col_names = ["Task 1", "Task 2", "Task 3", "Task 4", "task 5"]
# Save as csv
# np.savetxt(after_task_path + "_mAP.csv", after_mAP, delimiter=",", header=",".join(col_names), comments="", fmt="%s")

# For lowerbound save as csv
np.savetxt(after_task_path + "mAP.csv", after_mAP, delimiter=",", header=",".join(col_names), comments="", fmt="%s")

breakpoint()
# Plot average mAP using matplotlib and save to outputs/upperbound_num_epochs-8_bs16_avg_mAP.png
import matplotlib.pyplot as plt
plt.plot(avg_mAP, label='Average mAP', color='blue')
plt.savefig('./outputs/upperbound_num_epochs-8_bs16_avg_mAP.png')

# Show the mean of 10 last mAP values
print("Mean of 10 last mAP values:", np.mean(mAP[-10:]))

breakpoint()
