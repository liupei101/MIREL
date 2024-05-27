# -*- coding: utf-8 -*-
# used for collecting results from the CSV file exported from wandb
# > python3 collect_results_from_wandb.py CSV_FILE_EXPORTED_FROM_WANDB.csv std
import sys
import pandas as pd

path_csv = sys.argv[1]
print(f"Collecting results from {path_csv}")

mode = sys.argv[2]
print(f"run in mode = {mode}")
assert mode in ['std', 'more']

data = pd.read_csv(path_csv)

grouped_entries = dict()
for i, entry in enumerate(data['Name']):
    iloc = entry.find("-seed")
    
    if iloc < 0:
        print(f"Invalid entry: {entry}")

    run_name = entry[:iloc] + entry[(iloc + 8):]
    if run_name not in grouped_entries:
        grouped_entries[run_name] = [i]
    else:
        grouped_entries[run_name].append(i)

if mode == 'std':
    columns_to_read = ["Name", "bestckpt/train/test/pred/acc@mid", "bestckpt/train/test/pred/f1_score@mid", "bestckpt/train/test/pred/auc",
        "bestckpt/train/test/pred/IDConf/auc_conf_max_alpha", "bestckpt/train/test/pred/IDConf/aupr_conf_max_alpha", 
        "bestckpt/train/test/pred/OODDet/auc_det_alpha0", "bestckpt/train/test/pred/OODDet/aupr_det_alpha0", 
        "bestckpt/train/test/pred_ins/acc@mid", "bestckpt/train/test/pred_ins/f1_score@mid", "bestckpt/train/test/pred_ins/auc", 
        "bestckpt/train/test/pred_ins/IDConf/auc_conf_max_alpha", "bestckpt/train/test/pred_ins/IDConf/aupr_conf_max_alpha", 
        "bestckpt/train/test/pred_ins/OODDet/auc_det_alpha0", "bestckpt/train/test/pred_ins/OODDet/aupr_det_alpha0"
    ]
    special_col = [columns_to_read[_] for _ in [6, 7, 13, 14]]
else:
    columns_to_read = ["Name", "bestckpt/train/test/pred/acc@mid", "bestckpt/train/test/pred/f1_score@mid", "bestckpt/train/test/pred/auc",
        "bestckpt/train/test/pred/IDConf/auc_conf_max_alpha", "bestckpt/train/test/pred/IDConf/aupr_conf_max_alpha", 
        "bestckpt/train/test/pred/OODDet/auc_det_alpha0", "bestckpt/train/test/pred/OODDet/aupr_det_alpha0", 
        "bestckpt/train/test/pred_ins/acc@mid", "bestckpt/train/test/pred_ins/f1_score@mid", "bestckpt/train/test/pred_ins/auc", 
        "bestckpt/train/test/pred_ins/IDConf/auc_conf_max_alpha", "bestckpt/train/test/pred_ins/IDConf/aupr_conf_max_alpha", 
        "bestckpt/train/test/pred_ins/OODDet/auc_det_alpha0", "bestckpt/train/test/pred_ins/OODDet/aupr_det_alpha0",
        "bestckpt/train/test/pred/IDConf/mean_conf_max_prob", "bestckpt/train/test/pred/IDConf/mean_conf_exp_ent", 
        "bestckpt/train/test/pred/IDConf/mean_conf_alpha0", "bestckpt/train/test/pred/IDConf/mean_conf_mi", 
        "bestckpt/train/test/pred/OODDet/mean_det_max_prob", "bestckpt/train/test/pred/OODDet/mean_det_exp_ent",
        "bestckpt/train/test/pred/OODDet/mean_det_alpha0", "bestckpt/train/test/pred/OODDet/mean_det_mi",
        "bestckpt/train/test/pred_ins/IDConf/mean_conf_max_prob", "bestckpt/train/test/pred_ins/IDConf/mean_conf_exp_ent", 
        "bestckpt/train/test/pred_ins/IDConf/mean_conf_alpha0", "bestckpt/train/test/pred_ins/IDConf/mean_conf_mi", 
        "bestckpt/train/test/pred_ins/OODDet/mean_det_max_prob", "bestckpt/train/test/pred_ins/OODDet/mean_det_exp_ent", 
        "bestckpt/train/test/pred_ins/OODDet/mean_det_alpha0", "bestckpt/train/test/pred_ins/OODDet/mean_det_mi"
    ]
    special_col = [columns_to_read[_] for _ in [6, 7, 13, 14]]

ret_df = pd.DataFrame(columns=columns_to_read)

all_run_names = [k for k in grouped_entries.keys()]
all_run_names.sort()
for k in all_run_names:
    if len(grouped_entries[k]) != 5:
        print(f"Invalid run_name: {k}")

    sel_idxs = grouped_entries[k]
    sub_table = data.iloc[sel_idxs]

    for col in special_col:
        if sub_table[col].sum() <= 1e-6:
            from_col = col.replace("alpha0", "max_prob") 
            sub_table[col] = sub_table[from_col]

    stat_row_mean = {"Name": "Mean-" + k}
    stat_row_std  = {"Name": "Std-" + k}
    empty_row = {"Name": ""}
    for col in columns_to_read[1:]:
        stat_row_mean[col] = sub_table[col].mean()
        stat_row_std[col]  = sub_table[col].std(ddof=0)
        empty_row[col] = ""

    ret_df = ret_df.append(sub_table[columns_to_read], ignore_index=True)
    ret_df = ret_df.append(stat_row_mean, ignore_index=True)
    ret_df = ret_df.append(stat_row_std, ignore_index=True)
    ret_df = ret_df.append(empty_row, ignore_index=True)

ret_df.to_excel("temp.xlsx", index=False)
