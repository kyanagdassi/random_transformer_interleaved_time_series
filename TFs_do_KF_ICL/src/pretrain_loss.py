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
import time
from linalg_helpers import print_matrix
from haystack_plots import comp_quartiles



def gen_cong_lsts(config, model_name):

    output_dir, ckpt_dir, experiment_name = set_config_params(config, model_name)
    ckpt_steps = gen_ckpt_pred_steps(model_name)

    datasources = ["val", "train"]
    tf_avg_cong = []
    tf_std_cong = []
    train_exs_cong = []
    for datasource in datasources:
        train_exs = []
        tf_avg_lst = []
        tf_std_lst = []

        for ckpt in np.arange(1000, ckpt_steps[-1] + 1000, 1000):

            train_ex = ckpt*len(config.devices)*config.batch_size

            path = f"{output_dir}/prediction_errors{config.C_dist}_step={ckpt}.ckpt/train_conv_mult_cut_{datasource}_ortho_haar_state_dim_5_err_lss_examples.pkl"

            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    tf_errs = data["MOP"][0]

                    print(f"tf_errs.shape: {tf_errs.shape}")

                    #set all np.inf to 0 in tf_errs and zero_errs
                    tf_errs[np.isinf(tf_errs)] = 0


            except FileNotFoundError:
                print(f"ckpt {ckpt} not found")
                continue

            print(f"\n\n\nckpt: {ckpt}")
            tf_errs_mean_trial = np.mean(tf_errs, axis=1)
            print(f"tf_errs_mean_trial.shape: {tf_errs_mean_trial.shape}")

            tf_errs_mean = np.mean(tf_errs_mean_trial, axis=0)
            print(f"tf_errs_mean.shape: {tf_errs_mean.shape}")

            tf_avg = np.mean(tf_errs_mean, axis=0)
            tf_std = np.std(tf_errs_mean, axis=0) / np.sqrt(tf_errs_mean.shape[0])

            tf_avg_lst.append(tf_avg)
            tf_std_lst.append(tf_std)
            train_exs.append(train_ex)

        tf_avg_cong.append(tf_avg_lst)
        tf_std_cong.append(tf_std_lst)
        train_exs_cong.append(train_exs)

    return tf_avg_cong, tf_std_cong, train_exs_cong, output_dir


def get_multi_sys_ys(datasource, hay_len=5, ny=5, interleave_style="pretrain"):

    print(f"interleave_style: {interleave_style}")

    pre_path = f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/{datasource}_"
    if interleave_style == "pretrain":
        path = f"{pre_path}interleaved_traces_ortho_haar_ident_C_multi_cut.pkl"
    elif interleave_style == "needle_in_haystack":
        path = f"{pre_path}irrelevant_tokens_new_hay_insert_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl"
    elif interleave_style == "zero_cut":
        path = f"{pre_path}zero_cut_interleaved_traces_ortho_haar_ident_C_multi_cut.pkl"
    elif interleave_style == "backstory_train":
        path = f"{pre_path}interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)
        print(f"data.keys(): {data.keys()}")
        multi_sys_ys = data["multi_sys_ys"]
        print(f"path: {path}")
        print(f"multi_sys_ys.shape: {multi_sys_ys.shape}")
        if interleave_style == "pretrain":
            multi_sys_ys = multi_sys_ys[0] #get the first config's ys

        print(f"multi_sys_ys.shape: {multi_sys_ys.shape}")
        multi_sys_ys = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-1] - ny, multi_sys_ys.shape[-1]), axis=-1) #get the true test observations

        print(f"multi_sys_ys.shape: {multi_sys_ys.shape}")

        if interleave_style == "needle_in_haystack" or interleave_style == "zero_cut" or interleave_style =="backstory_train":
            raise Exception(f"in wrong conditional")
            #flatten dim 0 and dim 1 of multi_sys_ys
            multi_sys_ys = multi_sys_ys.reshape(multi_sys_ys.shape[0] * multi_sys_ys.shape[1],  multi_sys_ys.shape[2], multi_sys_ys.shape[3], multi_sys_ys.shape[4]) # (num_examples, num_trials, segment_length, ny)
        

        sys_choices_per_config = data["sys_choices_per_config"]
        seg_starts_per_config = data["seg_starts_per_config"]
        sys_inds_per_config = data["sys_inds_per_config"]
        real_seg_lens_per_config = data["real_seg_lens_per_config"]
        sys_dict_per_config = data["sys_dict_per_config"]
        
        if interleave_style == "pretrain":
            print(f"in right conditional")
            sys_choices_per_config = sys_choices_per_config[0]
            seg_starts_per_config = seg_starts_per_config[0]
            sys_inds_per_config = sys_inds_per_config[0]
            real_seg_lens_per_config = real_seg_lens_per_config[0]
            sys_dict_per_config = sys_dict_per_config[0]

        print(f"multi_sys_ys.shape: {multi_sys_ys.shape}")
        print(f"len(sys_choices_per_config): {len(sys_choices_per_config)}")
        print(f"len(seg_starts_per_config): {len(seg_starts_per_config)}")
        print(f"len(sys_inds_per_config): {len(sys_inds_per_config)}")
        print(f"len(real_seg_lens_per_config): {len(real_seg_lens_per_config)}")
    
    return multi_sys_ys, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, real_seg_lens_per_config, sys_dict_per_config


def get_multi_sys_ys_needle_in_haystack(datasource, hay_len=5, ny=5):
    #load the interleaved traces val data
    path = f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/{datasource}_irrelevant_tokens_new_hay_insert_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl"

    with open(path, 'rb') as f:
        data = pickle.load(f)
        print(f"data.keys(): {data.keys()}")
        multi_sys_ys = data["multi_sys_ys"]
        multi_sys_ys = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-1] - ny, multi_sys_ys.shape[-1]), axis=-1) #get the true test observations
        #flatten dim 0 and dim 1 of multi_sys_ys
        multi_sys_ys = multi_sys_ys.reshape(multi_sys_ys.shape[0] * multi_sys_ys.shape[1], multi_sys_ys.shape[2], multi_sys_ys.shape[3], multi_sys_ys.shape[4]) # (num_examples, num_trials, segment_length, ny)
        
        sys_choices_per_config = data["sys_choices_per_config"]
        seg_starts_per_config = data["seg_starts_per_config"]
        sys_inds_per_config = data["sys_inds_per_config"]
        real_seg_lens_per_config = data["real_seg_lens_per_config"]
        sys_dict_per_config = data["sys_dict_per_config"]

        print(f"multi_sys_ys.shape: {multi_sys_ys.shape}")
        print(f"len(sys_choices_per_config): {len(sys_choices_per_config)}")
        print(f"len(seg_starts_per_config): {len(seg_starts_per_config)}")
        print(f"len(sys_inds_per_config): {len(sys_inds_per_config)}")
        print(f"len(real_seg_lens_per_config): {len(real_seg_lens_per_config)}")

    return multi_sys_ys, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, real_seg_lens_per_config, sys_dict_per_config



def pseudo_prediction(history):

    # print_matrix(history, "history")
    inds = history.shape[1]
    # print(f"inds: {inds}")

    leftmat = history[:, 1:inds]
    # print(f"leftmat.shape: {leftmat.shape}")
    # print_matrix(leftmat, "leftmat")
    rightmat = history[:, 0:inds-1]
    # print(f"rightmat.shape: {rightmat.shape}")
    # print_matrix(rightmat, "rightmat")

    Uhat = leftmat @ np.linalg.pinv(rightmat)
    # print(f"Uhat.shape: {Uhat.shape}")

    pred = Uhat @ history[:,-1]
    # print(f"pred.shape: {pred.shape}")
    # print(f"pred: {pred}")
    return pred


def compute_pseudo_pred_errs(multi_sys_ys, seg_starts_per_config, real_seg_lens_per_config, sys_choices_per_config):
    pseudo_pred_errs = np.zeros_like(multi_sys_ys[:,:,:,0])
    print(f"pseudo_pred_errs.shape: {pseudo_pred_errs.shape}")

    for conf in range(multi_sys_ys.shape[0]):

        # print(f"sys_choices_per_config[conf]: {sys_choices_per_config[conf]}")

        errs_conf = np.zeros_like(multi_sys_ys[conf,:,:,0])
        # print(f"errs_conf.shape: {errs_conf.shape}")

        sys_init_ind_dict = {} # a dictionary that holds the system indices and how many initial indices have been seen for the config

        sys_appear = [] #holds the system indices that appear in the config

        seg_count = 0
        for seg in seg_starts_per_config[conf]:
            
            current_sys = sys_choices_per_config[conf][seg_count]
            # print(f"\n\ncurrent_sys: {current_sys}")
            if current_sys not in sys_appear: # if the system has not appeared before
                start_ind = seg + 1
                # print(f"start_ind: {start_ind}")
                real_seg_len = real_seg_lens_per_config[conf][seg_count]
                # print(f"real_seg_len: {real_seg_len}")
                end_ind = start_ind + real_seg_len
                # print(f"end_ind: {end_ind}")
                segment = multi_sys_ys[conf][:, start_ind:end_ind, :]
                # print(f"segment.shape: {segment.shape}")
                # print(f"segment: {segment}\n")
                sys_init_ind_dict[current_sys] = segment # add the system to the dictionary with the segment
                sys_appear.append(current_sys) #append the system to the list of systems that have appeared
                
                if start_ind < multi_sys_ys[conf].shape[1]: # check if start_ind is within the bounds of the multi_sys_ys array
                    true = multi_sys_ys[conf][0, start_ind, :]
                    errs_conf[0, start_ind] = np.linalg.norm(true)**2 # the 1-after initial prediction is hard-coded to zero

                for ys_ind in range(start_ind+1, end_ind): #generate the pseudo prediction for each ys_ind in the segment and compute the squared error
                    hist = multi_sys_ys[conf][0, start_ind:ys_ind, :].T

                    pred = pseudo_prediction(hist)
                    # print(f"pred: {pred}")
                    # print(f'pred.shape: {pred.shape}')
                    # print(f"multi_sys_ys[conf][0, ys_ind, :]: {multi_sys_ys[conf][0, ys_ind, :]}")
                    true = multi_sys_ys[conf][0, ys_ind, :]
                    
                    errs_conf[0, ys_ind] = np.linalg.norm(pred - true)**2

                    # print(f"errs_conf[0, {ys_ind}, :]: {errs_conf[0, ys_ind, :]}")
                    # print(f"ys_ind: {ys_ind}")
                    # print(f"hist.shape: {hist.shape}")

            elif sys_init_ind_dict[current_sys].shape[1] < 6: # still need to see 6 examples of the system
                
                old_seg_len = sys_init_ind_dict[current_sys].shape[1]
                # print(f"old segment.shape: {sys_init_ind_dict[current_sys].shape}")
                start_ind = seg + 1
                # print(f"start_ind: {start_ind}")
                real_seg_len = real_seg_lens_per_config[conf][seg_count]
                # print(f"real_seg_len: {real_seg_len}")
                end_ind = start_ind + real_seg_len
                # print(f"end_ind: {end_ind}")
                segment = multi_sys_ys[conf][:, start_ind:end_ind, :]
                # print(f"new segment.shape: {segment.shape}")

                segment = np.concatenate((sys_init_ind_dict[current_sys], segment), axis=1) # concatenate the new segment with the old segment

                # print(f"segment.shape: {segment.shape}")
                # print(f"segment: {segment}\n")
                sys_init_ind_dict[current_sys] = segment # add the system to the dictionary with the segment
                sys_appear.append(current_sys) #append the system to the list of systems that have appeared

                hist_count = 1
                for ys_ind in range(start_ind+1, end_ind): #generate the pseudo prediction for each ys_ind in the segment and compute the squared error
                    hist = segment[0, 0:old_seg_len + hist_count, :].T

                    pred = pseudo_prediction(hist)
                    # print(f"pred: {pred}")
                    # print(f'pred.shape: {pred.shape}')
                    # print(f"multi_sys_ys[conf][0, ys_ind, :]: {multi_sys_ys[conf][0, ys_ind, :]}")
                    true = multi_sys_ys[conf][0, ys_ind, :]
                    
                    errs_conf[0, ys_ind] = np.linalg.norm(pred - true)**2

                    # print(f"errs_conf[0, {ys_ind}, :]: {errs_conf[0, ys_ind]}")
                    # print(f"ys_ind: {ys_ind}")
                    hist_count += 1

            else: # have already seen 6 examples of the system
                pass

            seg_count += 1
        
        pseudo_pred_errs[conf] = errs_conf

    return pseudo_pred_errs

def compute_pseudo_pred_errs_needle_in_haystack(multi_sys_ys, seg_starts_per_config, real_seg_lens_per_config, sys_choices_per_config, zero_cut=False):
    pseudo_pred_errs = np.zeros_like(multi_sys_ys[:,:,:,0])
    print(f"pseudo_pred_errs.shape: {pseudo_pred_errs.shape}")

    for conf in range(multi_sys_ys.shape[0]):

        # print(f"sys_choices_per_config[conf]: {sys_choices_per_config[conf]}")

        errs_conf = np.zeros_like(multi_sys_ys[conf,:,:,0])
        # print(f"errs_conf.shape: {errs_conf.shape}")

        for trial in range(multi_sys_ys.shape[1]): # loop through each trial in the config
            sys_init_ind_dict = {} # a dictionary that holds the system indices and how many initial indices have been seen for the config

            sys_appear = [] #holds the system indices that appear in the config

            seg_count = 0
            if zero_cut:
                seg_starts = seg_starts_per_config[0][0] # get the first config's seg_starts
                sys_choices = sys_choices_per_config[0][0] # get the first config's sys_choices
                real_seg_lens = real_seg_lens_per_config[0][0] # get the first config's real_seg_lens

            else:
                seg_starts = seg_starts_per_config[conf][0]
                sys_choices = sys_choices_per_config[conf][0]
                real_seg_lens = real_seg_lens_per_config[conf][0]

            for seg in seg_starts:
                
                current_sys = sys_choices[seg_count]
                # print(f"\n\ncurrent_sys: {current_sys}")
                if current_sys not in sys_appear: # if the system has not appeared before
                    start_ind = seg + 1
                    real_seg_len = real_seg_lens[seg_count]
                    # print(f"real_seg_len: {real_seg_len}")
                    end_ind = start_ind + real_seg_len
                    # print(f"end_ind: {end_ind}")
                    segment = multi_sys_ys[conf][trial, start_ind:end_ind, :]
                    # print(f"segment.shape: {segment.shape}")
                    # print(f"segment.shape: {segment.shape}")
                    # print(f"segment: {segment}\n")
                    sys_init_ind_dict[current_sys] = segment # add the system to the dictionary with the segment
                    sys_appear.append(current_sys) #append the system to the list of systems that have appeared

                    # print("multi_sys_ys.shape: ", multi_sys_ys.shape)
                    # print(f"conf: {conf}, trial: {trial}, start_ind: {start_ind}, end_ind: {end_ind}")

                    true = multi_sys_ys[conf][trial, start_ind, :]
                    
                    errs_conf[trial, start_ind] = np.linalg.norm(true)**2 # the 1-after initial prediction is hard-coded to zero

                    for ys_ind in range(start_ind+1, end_ind): #generate the pseudo prediction for each ys_ind in the segment and compute the squared error

                        # print(f"\n\n\nys_ind: {ys_ind}")
                        hist = multi_sys_ys[conf][trial, start_ind:ys_ind, :].T

                        pred = pseudo_prediction(hist)
                        # print(f"pred: {pred}")
                        # print(f'pred.shape: {pred.shape}')
                        # print(f"multi_sys_ys[conf][0, ys_ind, :]: {multi_sys_ys[conf][0, ys_ind, :]}")
                        true = multi_sys_ys[conf][trial, ys_ind, :]
                        
                        errs_conf[trial, ys_ind] = np.linalg.norm(pred - true)**2

                        # print(f"errs_conf[0, {ys_ind}, :]: {errs_conf[0, ys_ind, :]}")
                        # print(f"ys_ind: {ys_ind}")
                        # print(f"hist.shape: {hist.shape}")

                elif sys_init_ind_dict[current_sys].shape[0] < 6: # still need to see 6 examples of the system

                    print(f"\n\n\nsys_init_ind_dict[current_sys].shape: {sys_init_ind_dict[current_sys].shape}")
                    
                    old_seg_len = sys_init_ind_dict[current_sys].shape[0]
                    # print(f"old segment.shape: {sys_init_ind_dict[current_sys].shape}")
                    start_ind = seg + 1
                    # print(f"start_ind: {start_ind}")
                    real_seg_len = real_seg_lens[seg_count]
                    # print(f"real_seg_len: {real_seg_len}")
                    end_ind = start_ind + real_seg_len
                    # print(f"end_ind: {end_ind}")
                    segment = multi_sys_ys[conf][trial, start_ind:end_ind, :]
                    # print(f"new segment.shape: {segment.shape}")

                    segment = np.concatenate((sys_init_ind_dict[current_sys], segment), axis=0) # concatenate the new segment with the old segment

                    # print(f"segment.shape: {segment.shape}")
                    # print(f"segment: {segment}\n")
                    sys_init_ind_dict[current_sys] = segment # add the system to the dictionary with the segment
                    sys_appear.append(current_sys) #append the system to the list of systems that have appeared

                    hist_count = 1
                    for ys_ind in range(start_ind+1, end_ind): #generate the pseudo prediction for each ys_ind in the segment and compute the squared error
                        hist = segment[0:old_seg_len + hist_count, :].T

                        print(f"hist.shape: {hist.shape}")

                        pred = pseudo_prediction(hist)
                        # print(f"hist_count: {hist_count}, pred: {pred}")
                        # print(f'pred.shape: {pred.shape}')
                        # print(f"multi_sys_ys[conf][0, ys_ind, :]: {multi_sys_ys[conf][0, ys_ind, :]}")
                        true = multi_sys_ys[conf][trial, ys_ind, :]
                        
                        errs_conf[trial, ys_ind] = np.linalg.norm(pred - true)**2

                        print(f"hist_count: {hist_count}, errs_conf[{trial}, {ys_ind}]: {errs_conf[trial, ys_ind]}, pred: {pred}, true: {true}")

                        # print(f"errs_conf[0, {ys_ind}, :]: {errs_conf[0, ys_ind]}")
                        # print(f"ys_ind: {ys_ind}")
                        hist_count += 1

                else: # have already seen 6 examples of the system
                    pass

                seg_count += 1
        
        pseudo_pred_errs[conf] = errs_conf

    return pseudo_pred_errs


def compute_avg_std(errs):
    print(f"errs.shape: {errs.shape}")

    errs_mean_trial = np.mean(errs, axis=1)
    print(f"errs_mean.shape: {errs_mean_trial.shape}")
    errs_mean = np.mean(errs_mean_trial, axis=0)
    print(f"errs_mean.shape: {errs_mean.shape}")
    avg = np.mean(errs_mean, axis=0)
    print(f"avg.shape: {avg.shape}")
    std = np.std(errs_mean, axis=0) / np.sqrt(errs_mean.shape[0])
    print(f"std.shape: {std.shape}")
    return avg, std

def save_pseudo_pred_medians(config, seg_starts_per_config, pseudo_pred_errs, steps_in = np.arange(1, 9), haystack_len=5, ex=0):
    pseudo_pred_errs = np.expand_dims(pseudo_pred_errs, axis=1)
    pseudo_err_exs = {"Pseudo": pseudo_pred_errs}
    pseudo_quartiles = comp_quartiles(config, pseudo_err_exs)

    seg = haystack_len

    fin_pseudo_pred_med_values = {}
    for step in steps_in:
        ind = ind = seg_starts_per_config[ex][0][seg] + step
        print(f"shape of pseudo_quartiles['Pseudo']: {pseudo_quartiles['Pseudo'].shape}")
        med_value = pseudo_quartiles["Pseudo"][1, 0, ind] #get the median value for the step in the segment
        print(f"step: {step}, med_value: {med_value}")
        fin_pseudo_pred_med_values[step] = med_value

    #save the fin_pseudo_pred_med_values to a pkl file (make path more general later)
    os.makedirs(f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/", exist_ok=True)
    with open(f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/val_irrelevant_tokens_new_hay_insert_pseudo_pred_medians_ortho_haar_ident_C_haystack_len_{haystack_len}.pkl", 'wb') as f:
        pickle.dump(fin_pseudo_pred_med_values, f)
        print(f"Saved fin_pseudo_pred_med_values")

    return fin_pseudo_pred_med_values



def compute_pseudo_pred_avg_pipeline(datasource):
    multi_sys_ys, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, real_seg_lens_per_config, sys_dict_per_config = get_multi_sys_ys(datasource)

    pseudo_pred_errs = compute_pseudo_pred_errs(multi_sys_ys, seg_starts_per_config, real_seg_lens_per_config, sys_choices_per_config)
    pseudo_pred_avg, pseudo_pred_std = compute_avg_std(pseudo_pred_errs)
    return pseudo_pred_avg, pseudo_pred_std


def format_scientific(x):
    # Format to scientific notation
    s = f"{x:.0e}"
    # Remove leading zeros in the exponent part
    return s.replace('e-0', 'e-').replace('e+0', 'e+')

def plot_haystack_train_conv_pretrain_x_axis(config, colors, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values_orig, train_exs_cong, tf_avg_cong, matching_indices, haystack_len, experiment, steps, nope, abs_err=False, finals=True, fig=None, ax=None, model_count=None, size=None, restart=False, fin_pseudo_pred_med_values=None, only_init=False, pseudo_pred_avg=None):


    # if not restart and only_init:
    #     return None

    markers = ['.','x', '>']
    

    #set x_values to be train_exs_cong[0] at the matching indices
    x_values = []
    for i in range(len(matching_indices)):
        x_values.append(tf_avg_cong[0][matching_indices[i]])

    #is x_values_orig the same length as x_values?
    if len(x_values_orig) != len(x_values):
        print(f"len x_values_orig: {len(x_values_orig)}")
        print(f"len x_values: {len(x_values)}")
        raise ValueError("x_values_orig and x_values are not the same length")

    valA = config.val_dataset_typ
    print(f"\n\nlen(x_values): {len(x_values)}")

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
        fig_len, ax_len = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
        multi_model = False
    else:
        multi_model = True
        if not restart:
            steps = [1,2]

    early_stop_ind = None

    if len(steps) > len(colors):
        # generate more colors from viridis colormap
        colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(steps)))

    print(f"\n\n in haystack train conv plot valA: {valA}, abs_err: {abs_err}\n\n")

    for key in fin_quartiles_ckpt.keys():

        beg_lab_suffix = f" into seg. 1" if config.irrelevant_tokens and config.new_hay_insert else " after initial"

        final_lab_suffix = f" into seg. {config.num_sys_haystack + 1}" if config.irrelevant_tokens and config.new_hay_insert else " after final"

        if key == "MOP": #key == "OLS_analytical_ir_1" or key == "OLS_ir_1": #key == "MOP" or 
            col_count = 0
            for step in steps:

                if fig is None and ax is None:
                    col_ind = col_count
                else:
                    col_ind = model_count

                key_lab = "TF" if key == "MOP" else key
                qs = np.array(fin_quartiles_ckpt[key][step])
                qs = np.transpose(qs)

                if valA == "gaussA":
                    if not abs_err:
                        qs -= 1

                # if restart:
                #     print(f"qs before subtracting pseudo pred med values: {qs}")
                #     qs -= fin_pseudo_pred_med_values[step]
                #     print(f"qs after subtracting pseudo pred med values: {qs}")
                    

                # #if key contains OLS then repeat the values in qs to be the length of x_values
                # if "OLS" in key:
                #     print(f"key: {key} qs shape: {qs.shape}")
                #     qs = np.repeat(qs, len(x_values), axis=0)
                #     print(f"qs shape after repeat: {qs.shape}")

                if step == 2:
                    #find the index of the minimum of qs[1]
                    early_stop_ind = np.argmin(qs[1])
                    print(f"early_stop_ind: {early_stop_ind}, x_values[early_stop_ind]: {x_values[early_stop_ind]}")

                    # raise NotImplementedError("Check the early stop index")
                
                if finals:

                    if multi_model and not restart:
                        label = f"{size}: {step} after final"
                    elif multi_model and restart:
                        label = f"{size}: {step} steps into seg. {haystack_len + 1}"
                    else:
                        label = f"{key_lab}: {step}{final_lab_suffix}"

                    if config.n_embd == 192:
                        cutoff = 17
                    elif config.n_embd == 128:
                        cutoff = 14
                    else:
                        cutoff = 1

                    # cutoff = 1 
                    # 
                    if step != 2:
                        continue   


                    ax.scatter(x_values[:-cutoff], qs[1][:-cutoff], label=label, s=50,marker=markers[min(2,step - 1)] if size is not None else ".", zorder=5 if key == "MOP" else 0, color=colors[model_count])

                    if restart and config.n_embd == 96:
                        if np.isfinite(fin_pseudo_pred_med_values[step]):
                            ax.axhline(y=fin_pseudo_pred_med_values[step], color=colors[model_count], linestyle="--", linewidth=1.5, zorder=0)
                        else:
                            print(f"Warning: fin_pseudo_pred_med_values[{step}] is not finite: {fin_pseudo_pred_med_values[step]}")
                col_count += 1

    if config.n_embd == 128:
        #find the index in train_exs_cong[0] that is closest to 122000*config.batch_size
        closest_ind = np.argmin(np.abs(np.array(train_exs_cong[0]) - (122000*config.batch_size)))
        closest_tf_avg = tf_avg_cong[0][closest_ind]
        #make a vertical line of the closest_tf_avg
        if not only_init:
            if pseudo_pred_avg is None:
                ax.axvline(x=closest_tf_avg, color="red", linestyle="--", linewidth=1.5)
    elif config.n_embd == 192:
        if pseudo_pred_avg is not None:
            ax.axvline(x=pseudo_pred_avg, color="red", linestyle="--", linewidth=1.5)

    if not multi_model:
        ax.invert_xaxis()
    ax.set_xlabel("Pretraining Error", fontsize=14)
    ax.set_ylabel(f"Error " + (f"- Pseudoinv Baseline Med. Err at step {steps[0]}" if only_init else "") + ("Ratio" if valA == "gaussA" and not abs_err else ""), fontsize=12)
    # if finals:
    #     ax.set_yscale('log')
    # else:
    #     ax.set_yscale('linear')

    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.grid(True, which="both")
    # have x-axis tick labels that are 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2
    xticks = [9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_scientific(x) for x in xticks], fontsize=8)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.set_ylim([-1e-2, 1e0])
    
    ax.legend(fontsize=10, ncol=2)#, loc="lower left", columnspacing=0.4)

    #add the date and time to the filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    figure_dir = f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_cut/pretrain_x_axis/" + (f"{config.datasource}/" if config.datasource != "val" else "") + ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("irrelevant_tokens/" if config.irrelevant_tokens else "") + ("same_tokens/" if config.same_tokens else "") + ("paren_swap/" if config.paren_swap else "") 
    os.makedirs(figure_dir, exist_ok=True)

    fig.tight_layout()
    # fig_len.tight_layout()
    
    if multi_model:
        if not (model_count == 3):
            return early_stop_ind
        
    print("saving figure")
    plt_type = "2_after" #"1_after" #"4_into_init"
    fig.savefig(figure_dir + ("only_init_" if only_init else "") + ("backstory_" if config.backstory and config.mem_suppress else ("init_seg_" if config.init_seg and config.mem_suppress else "")) + ("masked_" if config.masking and config.mem_suppress else ("unmasked_" if not config.masking and config.mem_suppress else "")) + ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("irrelevant_tokens_" if config.irrelevant_tokens else "") + ("same_tokens_" if config.same_tokens else "")+ ("paren_swap_" if config.paren_swap else "") + ("zero_cut_" if config.zero_cut else "") + ("new_hay_insert_" if config.new_hay_insert else "") + (f"late_start_{config.late_start}_" if config.late_start is not None else "") + ("abs_err_" if abs_err else "") + f"{valA}_embd_dim_{config.n_embd}_train_conv_pretrain_x_axis_all_models_{plt_type}_haystack_len_{haystack_len}_{timestamp}_" + "linearscale.pdf", transparent=True, format="pdf")
    if model_count == 3:
        plt.show()
        
    
    # fig_len.savefig(figure_dir + ("backstory_" if config.backstory and config.mem_suppress else ("init_seg_" if config.init_seg and config.mem_suppress else "")) + ("masked_" if config.masking and config.mem_suppress else ("unmasked_" if not config.masking and config.mem_suppress else "")) + ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("irrelevant_tokens_" if config.irrelevant_tokens else "") + ("same_tokens_" if config.same_tokens else "")+ ("paren_swap_" if config.paren_swap else "") + ("zero_cut_" if config.zero_cut else "") + ("new_hay_insert_" if config.new_hay_insert else "") + (f"late_start_{config.late_start}_" if config.late_start is not None else "") + ("abs_err_" if abs_err else "") + f"{valA}_embd_dim_{config.n_embd}_train_conv_pretrain_x_axis_haystack_len_{haystack_len}_{timestamp}_linearscale.pdf", transparent=True, format="pdf")

    
    return early_stop_ind


if __name__ == "__main__":
    fig,ax = plt.subplots(1, 1, figsize=(6, 6))

    config = Config()

    colors = ['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']

    start = time.time()
    pseudo_pred_stats = {}
    for datasource in ["val", "train"]:
        if datasource == "val":
            label = "Test"
        else:
            label = "Train"

        pseudo_pred_file = f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar/{datasource}_multi_cut_pseudo_pred_errs_ortho_haar_ident_C.pkl"

        if not os.path.exists(pseudo_pred_file):

            print(f"Computing pseudo prediction stats for {label} dataset")
            pseudo_pred_avg, pseudo_pred_std = compute_pseudo_pred_avg_pipeline(datasource)
            
            pseudo_pred_stats[label] = (pseudo_pred_avg, pseudo_pred_std)
            #save the pseudo prediction stats to a pkl file
            with open(pseudo_pred_file, 'wb') as f:
                pickle.dump(pseudo_pred_stats[label], f)
                print(f"Saved pseudo prediction stats for {label} dataset to {pseudo_pred_file}")
        else:
            print(f"Loading pseudo prediction stats for {label} dataset from {pseudo_pred_file}")
            with open(pseudo_pred_file, 'rb') as f:
                pseudo_pred_stats[label] = pickle.load(f)

    print(f"pseudo_pred_stats: {pseudo_pred_stats}")
    
    end = time.time()
    print(f"Time taken to compute pseudo prediction stats: {(end - start)/60} minutes")

    model_names = ["ortho_haar_tiny", "ortho_haar_small", "ortho_haar_medium_single_gpu", "ortho_haar_big"]

    sizes = ["212K - Tiny", "701K - Small", "2.42M - Medium", "10.7M - Big"]
    model_count = 0

    markers = ["o", "x"]
    linestyles = ["-", "--"]
    for model_name in model_names:
        tf_avg_cong, tf_std_cong, train_exs_cong, output_dir = gen_cong_lsts(config, model_name)
        count = 0


        #capitalize the first letter of each datasource
        datasources = ["Test", "Train"]
        for tf_avg_lst, tf_std_lst, train_exs in zip(tf_avg_cong, tf_std_cong, train_exs_cong):

            print(f"count: {count}")
            tf_avg_arr = np.array(tf_avg_lst)
            tf_std_arr = np.array(tf_std_lst)
            train_exs_arr = np.array(train_exs)
            # print(f"tf_avg_lst: {tf_avg_lst}")
            ax.plot(train_exs_arr, tf_avg_arr, color=colors[model_count], linewidth=2, label=f"{sizes[model_count]}" if datasources[count] == "Test" else "", marker=markers[count], linestyle=linestyles[count], markersize=5, zorder=2-count)
            ax.fill_between(train_exs_arr, tf_avg_arr - tf_std_arr, tf_avg_arr + tf_std_arr, color=colors[model_count], alpha=0.2, zorder=0)


            if model_count == 2 and datasources[count] == "Test":
                #create a text file that will hold the tf_avg_arr values for each axhline
                path = f"{output_dir}/figures/multi_cut/pretrain_loss/{model_name}_axhline_values.txt"
                #get the paired colormap from matplotlib

                os.makedirs(os.path.dirname(path), exist_ok=True)

                print(f"len of train_exs_cong: {len(train_exs_cong[0])} {len(train_exs_cong[1])}")

                #find the index of the element in train_exs_arr that is closest to 2.5e7
                print(f"train_exs_arr: {train_exs_arr}")
                closest_ind = np.argmin(np.abs(train_exs_arr - 2.5e7))
                print(f"closest_ind: {closest_ind}, train_exs_arr[closest_ind]: {train_exs_arr[closest_ind]}")
                #plot a horizontal line at tf_avg_arr[closest_ind]
                ax.axhline(y=tf_avg_arr[closest_ind], color="darkgreen", linestyle="--", linewidth=1.5, zorder=0, label=f"__no_legend__")

                with open(path, 'a') as f:
                    f.write(f"symbolic recall: {tf_avg_arr[closest_ind]}\n")

                #find the index of the element in train_exs_arr that is closest to 1.2e7
                closest_ind = np.argmin(np.abs(train_exs_arr - 1.2e7))
                print(f"closest_ind: {closest_ind}, train_exs_arr[closest_ind]: {train_exs_arr[closest_ind]}")
                #plot a horizontal line at tf_avg_arr[closest_ind]
                ax.axhline(y=tf_avg_arr[closest_ind], color="limegreen", linestyle="--", linewidth=1.5, zorder=0, label=f"__no_legend__")

                with open(path, 'a') as f:
                    f.write(f"bayes recall: {tf_avg_arr[closest_ind]}\n")

                #find the index of the element in train_exs_arr that is closest to 2e6
                closest_ind = np.argmin(np.abs(train_exs_arr - 2e6))
                print(f"closest_ind: {closest_ind}, train_exs_arr[closest_ind]: {train_exs_arr[closest_ind]}")
                #plot a horizontal line at tf_avg_arr[closest_ind]
                ax.axhline(y=tf_avg_arr[closest_ind], color="darkgreen", linestyle="--", linewidth=1.5, zorder=0, label=f"__no_legend__")

                with open(path, 'a') as f:
                    f.write(f"restart icl: {tf_avg_arr[closest_ind]}\n")

                #find the closest to 4e7
                closest_ind = np.argmin(np.abs(train_exs_arr - 4e7))
                print(f"closest_ind: {closest_ind}, train_exs_arr[closest_ind]: {train_exs_arr[closest_ind]}")
                #plot a horizontal line at tf_avg_arr[closest_ind]
                ax.axhline(y=tf_avg_arr[closest_ind], color="limegreen", linestyle="--", linewidth=1.5, zorder=0, label=f"__no_legend__")

                with open(path, 'a') as f:
                    f.write(f"IWL: {tf_avg_arr[closest_ind]}\n")

            print(f"pseudo_pred_stats[datasources[count]][0]: {pseudo_pred_stats[datasources[count]][0]}")
            #plot the pseudo prediction error as a horizontal line
            count += 1

        if model_count == 2:
            train_exs_arr_for_pseudo = train_exs_arr

        model_count += 1


    ax.axhline(pseudo_pred_stats["Test"][0], color="black", linewidth=3, label=f"__no_legend__", linestyle = ":", zorder=100)




    ax.set_ylabel("Pretraining Squared Error", fontsize=16)
    ax.set_xlabel("# of Examples Seen so Far in Training", fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim([4e-2, 1e0])
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=16)
    #set minor yticks every 0.01
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #set gridlines
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    


    fig.tight_layout()
    time = datetime.now().strftime("%Y-%m-%d_%H")
    fig_path = f"{output_dir}/figures/multi_cut/pretrain_loss/{model_name}_train_val_loss_cong_{time}_logy.pdf"

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, format='pdf')
    plt.show()