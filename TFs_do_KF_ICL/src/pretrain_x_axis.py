import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Liberation Serif'

import logging
import pickle
from datetime import datetime
import os
from data_processing import gen_ckpt_steps
from conv_plots_funcs import get_seg_starts_per_config
import torch
import gc
from core.config import Config
from data_train import set_config_params, gen_ckpt_pred_steps
from get_last_checkpoint import split_path
from haystack_plots import load_quartiles_ckpt_files

from pretrain_loss import gen_cong_lsts, get_multi_sys_ys, pseudo_prediction, compute_pseudo_pred_errs, compute_pseudo_pred_avg_pipeline, format_scientific, plot_haystack_train_conv_pretrain_x_axis, compute_pseudo_pred_errs_needle_in_haystack, save_pseudo_pred_medians, get_multi_sys_ys_needle_in_haystack


if __name__ == "__main__":
    fig,ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.invert_xaxis()

    colors = ['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']


    model_names = ["ortho_haar_tiny", "ortho_haar_small", "ortho_haar_medium_single_gpu", "ortho_haar_big"]


    sizes = ["Tiny-212K", "Small-701K", "Medium-2.42M", "Big-10.7M"]
    model_count = 0


    haystack_len = 7

    steps_in = [1, 2, 3, 7, 8]

    late_start = None
    paren_swap = False
    same_tokens = False
    fix_needle = False
    opposite_ortho = False
    only_beg = False
    fix_needle =False
    datasource = "val"
    acc = False

    config = Config() # Assuming Config is a class that holds the configuration settings

    excess = True
    if excess:
        #get the pseudo prediction error medians
        path = f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/val_irrelevant_tokens_new_hay_insert_pseudo_pred_medians_ortho_haar_ident_C_haystack_len_{haystack_len}.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                fin_pseudo_pred_med_values = pickle.load(f)
            print(f"Loaded pseudo prediction medians from {path}")
        else:

            multi_sys_ys, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, real_seg_lens_per_config, sys_dict_per_config = get_multi_sys_ys_needle_in_haystack("val", haystack_len, ny=5)

            pseudo_pred_errs = compute_pseudo_pred_errs_needle_in_haystack(multi_sys_ys, seg_starts_per_config, real_seg_lens_per_config, sys_choices_per_config)

            fin_pseudo_pred_med_values = save_pseudo_pred_medians(config, seg_starts_per_config, pseudo_pred_errs, steps_in=np.arange(1, 5), haystack_len=haystack_len, ex=0)
    else:
        fin_pseudo_pred_med_values = None

    


    #save the pseudo prediction average to npy file
    pseudo_pred_avg_file = f"data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/pseudo_pred_avg_{datasource}.npy"

    print(os.path.exists(pseudo_pred_avg_file))

    if not os.path.exists(pseudo_pred_avg_file):
        pseudo_pred_avg, pseudo_pred_std = compute_pseudo_pred_avg_pipeline(datasource) #compute the pseudo prediction average and std
        os.makedirs(os.path.dirname(pseudo_pred_avg_file), exist_ok=True)  # Ensure the directory exists
        np.save(pseudo_pred_avg_file, pseudo_pred_avg)
        print(f"Saved pseudo_pred_avg to {pseudo_pred_avg_file}")
    else:
        #load the pseudo_pred_avg from the file
        pseudo_pred_avg = np.load(pseudo_pred_avg_file)
        print(f"Loaded pseudo_pred_avg from {pseudo_pred_avg_file}")



    for model_name in model_names:
        tf_avg_cong, tf_std_cong, train_exs_cong, output_dir = gen_cong_lsts(config, model_name) #get pretrain errors

        model_dir, experiment = split_path(output_dir)

        for i in np.arange(0,1):

            if i == 0:
                restart = False
                steps = steps_in
            else:
                restart = True
                steps = [4]
            irrelevant_tokens = restart
            new_hay_insert = restart
            config.override("datasource", datasource) # set the datasource in the config object
            config.override("acc", acc) # set the acc in the config object for using the ACCESS server

            # config.override("late_start", late_start) # set the late_start in the config object
            config.override("late_start", late_start)

            config.override("paren_swap", paren_swap) # set the paren_swap in the config object
            if config.paren_swap:
                print("Running paren swap experiment\n\n\n")

            config.override("same_tokens", same_tokens) # set the same_tokens in the config object
            if config.same_tokens:
                print("Running same tokens experiment\n\n\n")

            config.override("irrelevant_tokens", irrelevant_tokens) # set the irrelevant_tokens in the config object
            if config.irrelevant_tokens:
                print("Running irrelevant tokens experiment\n\n\n")

            config.override("fix_needle", fix_needle) # set the fix_needle in the config object
            if config.fix_needle:
                print("Running fix needle experiment\n\n\n")

            config.override("new_hay_insert", new_hay_insert) # set the new_hay_insert in the config object
            if config.new_hay_insert:
                print("Running new hay insertion experiment\n\n\n")

            config.override("opposite_ortho", opposite_ortho) # set the opposite_ortho in the config object
            if config.opposite_ortho:
                config.override("val_dataset_typ", "ortho")

            config.override("only_beg", only_beg) # set the only_beg in the config object
            if config.only_beg:
                print("only plotting the beginning evals\n\n\n")

            train_conv_fin_quartiles_file, train_conv_beg_quartiles_file, x_values_file, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values = load_quartiles_ckpt_files(config, haystack_len, model_dir, experiment, False)

            print(f"train_conv_fin_quartiles_file: {train_conv_fin_quartiles_file}")

            train_exs_cong_arr = np.array(train_exs_cong[0])


            # Safer approach to find matching indices between arrays
            matching_indices = []
            train_exs_values = []

            # For each value in x_values, try to find a match in train_exs_cong
            x_ind = 0
            
            x_inds = []
            for x in x_values:
                # Find where values match (within some small tolerance for floating point)
                matches = np.where(np.isclose(train_exs_cong_arr, x, rtol=1e-10))[0]
                
                if len(matches) > 0:
                    # If found, use the first match
                    matching_indices.append(matches[0])
                    train_exs_values.append(train_exs_cong_arr[matches[0]])
                    x_inds.append(x_ind)
                else:
                    print(f"No match found for x_value: {x}")

                x_ind += 1

            matching_indices = np.array(matching_indices)
            train_exs_values = np.array(train_exs_values)
            print(f"matching_indices: {matching_indices}")
            print(f"x_inds: {x_inds}")
            print(f"train_exs_values: {train_exs_values}")
            print(f"x_values: {x_values}")
            print(f"train_exs_cong_arr: {train_exs_cong_arr}")

            print(f"\n\nSTEPS: {steps}")

            plot_haystack_train_conv_pretrain_x_axis(config, colors, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values, train_exs_cong, tf_avg_cong, matching_indices, haystack_len, experiment, steps=steps, nope=False, abs_err=False, finals=True, fig=fig, ax=ax, model_count=model_count, size=sizes[model_count], restart=restart, fin_pseudo_pred_med_values=fin_pseudo_pred_med_values, only_init=False, pseudo_pred_avg=pseudo_pred_avg)
        model_count += 1