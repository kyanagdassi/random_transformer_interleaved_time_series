import matplotlib.pyplot as plt
#set font to liberation sans
plt.rcParams['font.family'] = 'Liberation Serif'
import numpy as np
from core import Config
from data_train import set_config_params
import os
import argparse
import time



def plot_init_seg_restart_train_conv(config, output_dir, hay_len, colors):
    x_values_path = f"{output_dir}/needles/train_conv/irrelevant_tokens_new_hay_insert_x_values_haystack_len_{hay_len}.npy"

    #load the x values npy file
    x_values = np.load(x_values_path)


    col_count = 0
    step = 1
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4.7))


    seg = "fin"
    file_len = hay_len

    data_path = f"{output_dir}/needles/train_conv/irrelevant_tokens_new_hay_insert_train_conv_{seg}_quartiles_haystack_len_{file_len}.pkl"

    x_values_path = f"{output_dir}/needles/train_conv/irrelevant_tokens_new_hay_insert_x_values_haystack_len_{file_len}.npy"

    #load the x values npy file
    x_values = np.load(x_values_path)

    with open(data_path, "rb") as f:
        fin_quartiles = np.load(f, allow_pickle=True)
        fin_quartiles = fin_quartiles["MOP"]

    if len(fin_quartiles.keys()) > len(colors):
        colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(fin_quartiles.keys())))

    for step in fin_quartiles.keys():
        qs = np.array(fin_quartiles[step])
        qs = np.transpose(qs)

        ax.plot(x_values, qs[1], label=f"{step} steps into seg. {hay_len+1}", markersize=5, marker=".", color=colors[col_count], linewidth=2)

        ax.fill_between(x_values, qs[0], qs[2], alpha=0.2, color=colors[col_count])
        col_count += 1

    ax.set_ylabel(f"Error", fontsize=12)
    ax.set_xlabel("# of Training Examples", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, which='both')


    # ax.set_ylim([-0.15, 1.1])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim([5e5, 1e8])
    ax.set_ylim([2e-3, 1.5])

    # ax.set_xticks(tks)
    ax.legend(loc="lower left", ncols = 2, columnspacing=0.2, handletextpad=0.5, labelspacing=0.2, fontsize=10)

    timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

    fig_path = f"{output_dir}/figures/restart_sys/train_conv/{config.val_dataset_typ}_n_embd_{config.n_embd}_restart_sys_train_conv_all_steps_into_seg_{hay_len+1}_log_{timestamp}.pdf"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, format='pdf')
    plt.show()

if __name__ == "__main__":
    hay_len = 2
    colors = ['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']
    parser = argparse.ArgumentParser(description="Plot initial segmentation restart training convergence")

    # model_name = "ortho_haar_medium_single_gpu"

    parser.add_argument("--model_name", type=str, default="ortho_haar_medium_single_gpu", help="Name of the model to use for training")

    args = parser.parse_args()
    model_name = args.model_name

    config = Config()
    output_dir, ckpt_dir, experiment_name = set_config_params(config, model_name)

    plot_init_seg_restart_train_conv(config, output_dir, hay_len, colors)
