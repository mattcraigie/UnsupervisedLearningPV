import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scattering_transform.filters import ClippedMorlet
from models import NFSTRegressor
import yaml
import argparse
import numpy as np
import pandas as pd
import shutil


def plot_filter_transformations(filters_final, filters_initial, save_dir, transform_fn, title, file_name):
    """
    Plot filter transformations.

    Parameters:
    - model: The model containing the filters.
    - save_dir: Directory to save the plots.
    - transform_fn: Function to transform the filters.
    - title: Title for the plots.
    - file_name: File name to save the plot.
    """
    num_scales = len(filters_final)

    fig, axes = plt.subplots(nrows=3, ncols=num_scales, figsize=(9, 9), dpi=100)

    for j in range(num_scales):
        filt_final = transform_fn(filters_final[j][0].cpu().detach())
        filt_initial = transform_fn(filters_initial[j][0].cpu().detach())

        filt_difference = filt_final - filt_initial
        max_abs_value = max(abs(filt_difference.min()), abs(filt_difference.max()))
        norm_difference = Normalize(vmin=-max_abs_value, vmax=max_abs_value)

        filt_asymmetry = filt_final.clone()
        filt_asymmetry[1:] = filt_asymmetry[1:] - filt_asymmetry[1:].flip(0)
        filt_asymmetry[0, :] = 0

        axes[0, j].imshow(filt_final)
        axes[1, j].imshow(filt_difference, norm=norm_difference, cmap='bwr')
        axes[2, j].imshow(filt_asymmetry, cmap='bwr')

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel('Learned Filter', fontsize=14)
    axes[1, 0].set_ylabel('Difference from \nMorlet Initialisation', fontsize=14)
    axes[2, 0].set_ylabel('Asymmetry in Filter', fontsize=14)

    for i in range(num_scales):
        axes[0, i].set_title(f'$j={i}$')

    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, file_name))


def final_filters_plot(filters_final, filters_initial, save_path, nfn_width):
    num_scales = len(filters_final)

    fig, axes = plt.subplots(nrows=3, ncols=num_scales, figsize=(9, 9), dpi=100)

    average_diffs = []

    for j in range(num_scales):
        filt_final = filters_final[j][0].cpu().detach()
        filt_initial = filters_initial[j][0].cpu().detach()
        filt_difference = filt_final - filt_initial
        max_abs_value = max(abs(filt_difference.min()), abs(filt_difference.max()))
        norm_difference = Normalize(vmin=-max_abs_value, vmax=max_abs_value)

        ft_of_diff = torch.fft.fft2(filt_difference)
        ft_of_diff[0, 0] = 0

        axes[1, 0].imshow(filt_final)
        axes[1, 0].imshow(filt_difference, norm=norm_difference, cmap='bwr')
        axes[2, j].imshow(np.fft.fftshift(np.abs(ft_of_diff)))

        average_diffs.append(filt_difference.abs().mean().item() / filt_initial.abs().mean().item())

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel('Learned Filter', fontsize=14)
    axes[1, 0].set_ylabel('Difference from\nMorlet Initialisation', fontsize=14)
    axes[2, 0].set_ylabel('Inverse Fourier Transform of\nDifference From\nMorlet Initialisation', fontsize=14)

    for i in range(num_scales):
        axes[0, i].set_title(f'$j={i}$')

    plt.suptitle("Neural Field Filters of NFST Model with NFN width {}".format(nfn_width))

    print('Average diffs: ', average_diffs)
    print('Mean over filters: ', np.mean(average_diffs))

    plt.savefig(save_path)
    plt.close()

    # NEXT: MAKE THIS RUN FOR ALL THE MODELS IN THE FOLDER



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # redo the above with shortened options
    parser.add_argument('-m', '--model_save_path', type=str, required=True)
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-s', '--save_dir', type=str, required=True)
    # parser.add_argument('-n', '--num_neurons', type=int, default=256, required=False)

    args = parser.parse_args()

    # load config yaml
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    all_folders = np.sort(os.listdir(args.model_save_path))  # not all are folders
    all_folders = [folder for folder in all_folders if os.path.isdir(os.path.join(args.model_save_path, folder))]

    all_nfst_sizes = pd.read_csv(os.path.join(args.model_save_path, 'summary.csv'), header=None)[0].values.astype(int)[1:]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for folder, nfst_size in zip(all_folders, all_nfst_sizes):

        current_read_folder = os.path.join(args.model_save_path, folder)
        current_output_folder = os.path.join(args.save_dir, folder)
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)


        print(nfst_size)
        config['analysis_kwargs']['model_kwargs']['subnet_hidden_sizes'] = [nfst_size, nfst_size]

        all_repeats = np.sort(os.listdir(current_read_folder))

        for repeat in all_repeats:
            print(folder)
            print(config['analysis_kwargs']['model_kwargs']['subnet_hidden_sizes'])
            model = NFSTRegressor(**config['analysis_kwargs']['model_kwargs'])
            model.load_state_dict(torch.load(os.path.join(current_read_folder, repeat, 'model.pt')))

            filters_final = [filt.clone() for filt in model.filters.filters]
            model.filters.load_state_dict(model.initial_filters_state)
            model.filters.update_filters()
            filters_initial = model.filters.filters

            out_file = os.path.join(current_output_folder, repeat + '_filters.png')

            final_filters_plot(filters_final, filters_initial, out_file, nfst_size)

    # tar gzip the entire folder
    shutil.make_archive(args.save_dir, 'gztar', args.save_dir)




