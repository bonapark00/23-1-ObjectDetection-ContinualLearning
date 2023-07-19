import numpy as np
import os
from calculate_auc import get_mAP_AUC

mode = "mir"
dataset = "shift"
# batchsize = 16
# temp_batchsize = 8
sd = 1

path = os.path.join("outputs", mode)
any_mAP = np.load(os.path.join(path, "mAP.npy"))
any_task_training = np.load(os.path.join(path, "task_training.npy"))
any_task_evaluating = np.load(os.path.join(path, "task_evaluating.npy"))
any_eval_time = np.load(os.path.join(path, "eval_time.npy"))

task_records = {
    "test_mAP": any_mAP,
    "task_training": any_task_training,
    "task_evaluating": any_task_evaluating,
    "data_cnt": any_eval_time
}

auc_dict = get_mAP_AUC(task_records, 4)
mAP_AUC = auc_dict["mAP_AUC"]
print("mAP_AUC:", mAP_AUC)
breakpoint()