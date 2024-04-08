import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from mocks import *

def losses_plot(root, techniques, test_type, save_dir):

    for technique in techniques:
        technique_dir = os.path.join(root, technique)

        sub_folders = []
        for filename in os.listdir(technique_dir):
            full_path = os.path.join(technique_dir, filename)  # Concatenating the directory and filename
            if os.path.isdir(full_path) and filename.startswith(test_type):
                sub_folders.append(full_path)

        sub_folders = np.sort(sub_folders)

        fig, axes = plt.subplots(ncols=len(sub_folders), nrows=2, figsize=(len(sub_folders) * 3, 4), sharey=False,
                                 dpi=300)
        plt.suptitle(technique + ' losses')

        for i, sub_folder in enumerate(sub_folders):
            repeat_folders = os.listdir(sub_folder)
            repeat_folders = np.sort(repeat_folders)

            for repeat_folder in repeat_folders:
                # need to decode utf-8 because this is reading in as byte strings??? not sure why
                loss_path = os.path.join(sub_folder, repeat_folder.decode("utf-8"), 'losses.npy')
                losses = np.load(loss_path)
                axes[0, i].plot(losses[0], c='blue', alpha=0.3, linewidth=0.5)
                axes[1, i].plot(losses[1], c='red', alpha=0.3, linewidth=0.5)
                # axes[i].set_xticks([])

                # break
            axes[0, i].set_title(sub_folder.split('/')[-1])
            axes[0, i].set_yticks([])
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel('Train Loss')
        axes[1, 0].set_ylabel('Val Loss')

        plt.savefig(os.path.join(save_dir, test_type + '_' + technique + '_losses.png'))


def plot_nfst_sizes_from_csvs(csv_paths, labels, plot_name, value='mean'):
    if len(csv_paths) != len(labels):
        raise ValueError("The number of CSV paths must match the number of labels.")

    plt.figure(figsize=(10, 6))

    colours = ['red', 'goldenrod', 'blue', 'pink']

    for csv_path, label, colour in zip(csv_paths, labels, colours):
        # Read the CSV into a DataFrame
        df = pd.read_csv(csv_path)

        # Assuming the first unnamed column represents sample sizes and is used as the x-axis
        x = df.iloc[:, 0]

        if value == 'mean':
            y = df['mean']
            error = df['std']
            # plt.errorbar(x, y, yerr=error, fmt='-o', capsize=5, color=colour)
            plt.plot(x, y, '-o', label=f'{label}', color=colour)
            plt.fill_between(x, y - error, y + error, alpha=0.1, color=colour)
        elif value == 'median':
            y = df['50%']  # Median
            lower_error = y - df['25%']
            upper_error = df['75%'] - y
            # plt.errorbar(x, y, yerr=[lower_error, upper_error], fmt='-o', capsize=5,
            #              label=f'{label}', color=colour)
            plt.fill_between(x, df['25%'], df['75%'], alpha=0.1, color=colour)
            plt.plot(x, y, '-o', label=f'{label}', color=colour)
        elif value == 'max':
            y = df['max']
            plt.plot(x, y, '-o', label=f'{label}', color=colour)

        else:
            raise ValueError("Invalid value option. Choose 'mean' or 'median'.")

    plt.xlabel('Number of Neurons')
    plt.ylabel(f'{value.capitalize()} $\\eta$')
    plt.title(f'{value.capitalize()} Parity Violation Detection Score')
    plt.legend()
    plt.grid(True)
    plt.semilogx()

    # a black line at y=3
    plt.axhline(y=3, color='black', linestyle='--')

    plt.savefig(f'{plot_name}_{value}.png')


def plot_datascaling_from_csvs(csv_paths, labels, plot_name, value='mean'):
    if len(csv_paths) != len(labels):
        raise ValueError("The number of CSV paths must match the number of labels.")

    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True)

    colours = ['red', 'goldenrod', 'blue', 'pink']

    for csv_path, label, colour in zip(csv_paths, labels, colours):
        # Read the CSV into a DataFrame
        df = pd.read_csv(csv_path)

        # Assuming the first unnamed column represents sample sizes and is used as the x-axis
        x = df.iloc[:, 0]

        if value == 'mean':
            y = df['mean']
            error = df['std']
            # plt.errorbar(x, y, yerr=error, fmt='-o', capsize=5, color=colour)
            ax1.plot(x, y, '-o', label=f'{label}', color=colour)
            ax1.fill_between(x, y - error, y + error, alpha=0.1, color=colour)

            ax2.plot(x, y, '-o', label=f'{label}', color=colour)
            ax2.fill_between(x, y - error, y + error, alpha=0.1, color=colour)
        elif value == 'median':
            y = df['50%']  # Median
            lower_error = y - df['25%']
            upper_error = df['75%'] - y
            # plt.errorbar(x, y, yerr=[lower_error, upper_error], fmt='-o', capsize=5,
            #              label=f'{label}', color=colour)
            ax1.fill_between(x, df['25%'], df['75%'], alpha=0.1, color=colour)
            ax1.plot(x, y, '-o', label=f'{label}', color=colour)

            ax2.fill_between(x, df['25%'], df['75%'], alpha=0.1, color=colour)
            ax2.plot(x, y, '-o', label=f'{label}', color=colour)
        elif value == 'max':
            y = df['max']
            plt.plot(x, y, '-o', label=f'{label}', color=colour)

        else:
            raise ValueError("Invalid value option. Choose 'mean' or 'median'.")

    plt.xlabel('Training Dataset Size')
    fig.text(0.04, 0.5, f'{value.capitalize()} $\\eta$', va='center', rotation='vertical')

    ax1.set_ylim(-3, 10)  # Limiting y-axis
    ax2.set_ylim(10, 80)  # Limiting y-axis

    # Adjust ax1 (bottom subplot) position
    pos1 = ax1.get_position()  # get the original position of ax1
    pos2 = ax2.get_position()  # get the original position of ax2
    height_adjusted = (pos1.height + pos2.height) / 2  # calculate the new height

    # Set new positions
    pos1_new = [pos1.x0, pos1.y0, pos1.width, height_adjusted]
    pos2_new = [pos2.x0, pos1.y0 + height_adjusted, pos2.width, height_adjusted]

    ax1.set_position(pos1_new)
    ax2.set_position(pos2_new)

    # Removing the spines and ticks of the bottom plot of ax2 to clean up the interface
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(labelbottom=False)

    ax1.legend()
    plt.semilogx()

    # a black line at y=3
    plt.axhline(y=3, color='black', linestyle='--')

    ax1.spines['top'].set_linestyle(':')  # Set the linestyle to dashed
    ax1.spines['top'].set_color('black')  # Set the color to grey
    ax1.spines['top'].set_alpha(0.3)  # Set the transparency for a faint appearance

    ax1.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10])
    ax1.set_yticklabels(['', -2, '', 2, '', 6, '', 10])

    ax2.set_yticks([20, 30, 40, 50, 60, 70, 80, 90])
    ax2.set_yticklabels(['', 30, '', 50, '', 70, '', 90])

    plt.savefig(f'{plot_name}_{value}.png')


def datascaling_plot():
    root = "//scratch/smp/uqmcrai4/parity/output/data_scaling"

    folders = ['nfst', 'mst', 'cnn_circ']

    all_csvs = []
    for folder in folders:
        all_csvs.append(os.path.join(root, folder, 'summary.csv'))

    labels = ['NFST', 'WST', 'CNN']
    save_path = '/clusterdata/uqmcrai4/UnsupervisedLearningPV/output/plots/data_scaling'

    plot_datascaling_from_csvs(all_csvs, labels, save_path, value='mean')
    plot_datascaling_from_csvs(all_csvs, labels, save_path, value='median')
    plot_datascaling_from_csvs(all_csvs, labels, save_path, value='max')

    save_path = '/clusterdata/uqmcrai4/UnsupervisedLearningPV/output/plots/'
    losses_plot(root, folders, 'data_scaling', save_path)


def nfst_sizes_plot():
    root = "//scratch/smp/uqmcrai4/parity/output/nfst_sizes"

    folders = ['nfst', 'nfst_noinit', 'nfst_symm']

    all_csvs = []
    for folder in folders:
        all_csvs.append(os.path.join(root, folder, 'summary.csv'))

    labels = ['Morlet Init.', 'Random Init.', 'Symmetric']
    save_path = '/clusterdata/uqmcrai4/UnsupervisedLearningPV/output/plots/nfst_sizes'

    plot_nfst_sizes_from_csvs(all_csvs, labels, save_path, value='mean')
    plot_nfst_sizes_from_csvs(all_csvs, labels, save_path, value='median')
    plot_nfst_sizes_from_csvs(all_csvs, labels, save_path, value='max')

    save_path = '/clusterdata/uqmcrai4/UnsupervisedLearningPV/output/plots/'
    losses_plot(root, folders, 'nfst_sizes', save_path)



def verification_plot(root, techniques, test_type, save_dir, colours=None):
    assert test_type in ['sensitivity', 'data_scaling'], 'test_type must be one of "sensitivity" or "datascaling"'

    key = 'ratio_left' if test_type == 'sensitivity' else 'num_train_val_mocks'
    csv_name = 'sensitivity.csv' if test_type == 'sensitivity' else 'data_scaling.csv'

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    if colours is None:
        colours = ['red', 'goldenrod', 'blue', 'pink']

    for colour, technique in zip(colours, techniques):
        df = pd.read_csv(os.path.join(root, test_type, technique, csv_name))
        print(df)
        axes[0].scatter(df[key], df['ks_test_pvalue'], c=colour)

        axes[1].scatter(df[key], df['bootstrap_std'] / df['verification_std'], c=colour)

    axes[0].set_title('Distribution Similarity Test')

    # shade between 0 and 0.1
    min_x, max_x = axes[0].get_xlim()
    axes[0].fill_between([min_x, max_x], 0, 0.1, color='grey', alpha=0.2)
    axes[0].axhline(0.1, c='black', linestyle=':')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xlim(min_x, max_x)

    axes[1].set_title('Std. Dev. Equivalence Test')
    axes[1].axhline(1)

    # shade between 0.8 and 0.9, and between 1.1 and 1.2. Also set lims to 0.8 and 1.2
    axes[1].fill_between([min_x, max_x], 0.8, 0.9, color='grey', alpha=0.2)
    axes[1].fill_between([min_x, max_x], 1.1, 1.2, color='grey', alpha=0.2)
    axes[1].set_ylim(0.8, 1.2)
    axes[1].axhline(0.9, c='black', linestyle=':')
    axes[1].axhline(1.1, c='black', linestyle=':')
    axes[1].set_xlim(min_x, max_x)

    if test_type == 'sensitivity':
        axes[0].set_xlabel('Ratio of Training to Test Data')
        axes[1].set_xlabel('Ratio of Training to Test Data')
    elif test_type == 'data_scaling':
        axes[0].set_xlabel('Number of Samples in Train/Val Set')
        axes[0].set_xlabel('Number of Samples in Train/Val Set')

    axes[0].set_ylabel('KS Test p-value')
    axes[1].set_ylabel('Bootstrap std / Verification std')

    plt.savefig(os.path.join(save_dir, test_type + '_verification.png'))


def plot_toy_data_patches():
    mocks = create_parity_violating_mocks_2d(3, 32, 15, 1, 4, 8)
    single_blue = create_parity_violating_mocks_2d(3, 32, 1, 1, 4, 8)
    # single_red = create_parity_violating_mocks_2d(3, 32, 1, 1, 4, 8)

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), dpi=300)
    for i, ax in enumerate(axes):
        ax.imshow(np.clip(mocks[i], 0, 1), cmap='Greys', vmax=2)

        # Convert single mock data to an RGBA image
        rgba_single = np.zeros((*single_blue[i].shape, 4))
        rgba_single[..., 2] = single_blue[i]  # Set blue channel
        rgba_single[..., 3] = single_blue[i]  # Set alpha channel based on the data value

        # Overplot the single in blue with variable opacity
        ax.imshow(rgba_single, vmax=1)

        # # do the same for red
        # rgba_single = np.zeros((*single_red[i].shape, 4))
        # rgba_single[..., 0] = single_red[i]  # Set red channel
        # rgba_single[..., 3] = single_red[i]  # Set alpha channel based on the data value
        #
        # ax.imshow(rgba_single, vmax=1)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('/clusterdata/uqmcrai4/UnsupervisedLearningPV/output/plots/toy_data_patches.png')


def plot_single_triangle():
    size = 32
    a, b = 4, 8

    grid = np.zeros((size, size), dtype=int)

    point_1 = np.array([16, 16])

    point_2 = point_1 + a * np.array([0, 1])
    point_3 = point_1 + b * np.array([-1, 0])

    for p in [point_1, point_2, point_3]:
        # Map the points to the nearest grid points
        p_grid = np.round(p).astype(int) % size
        np.add.at(grid, tuple(p_grid.T), 1)

    x = grid
    k = np.fft.fftshift(np.fft.fft2(grid))

    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    axes[0].imshow(x, cmap='Greys', vmax=1)
    axes[1].imshow(np.abs(k))
    plt.show()
    plt.savefig('/clusterdata/uqmcrai4/UnsupervisedLearningPV/output/plots/single_triangle.png')


if __name__ == '__main__':

    datascaling_plot()
    nfst_sizes_plot()
    plot_toy_data_patches()

    # # parse arguments for plotting
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--plot_func', type=str, default='performance')
    # parser.add_argument('--root', type=str, required=True)
    # parser.add_argument('--techniques', type=str, nargs='+', default=['nfst', 'nfst_mini', 'mst', 'cnn'])
    # parser.add_argument('--test_type', type=str, default='sensitivity')
    # parser.add_argument('--plot_type', type=str, default='max')
    # parser.add_argument('--save_dir', type=str, default='./output/plots/')
    # parser.add_argument('--colours', type=str, nargs='+', default=None)
    #
    # # also for the type of plotting function to call: losses, performance, verification
    #
    #
    # args = parser.parse_args()
    #
    # if args.plot_func == 'losses':
    #     losses_plot(args.root, args.techniques, args.test_type, args.save_dir)
    #
    # elif args.plot_func == 'performance':
    #     performance_plot(args.root, args.techniques, args.test_type, args.plot_type, args.save_dir, args.colours)
    #
    # elif args.plot_func == 'verification':
    #     verification_plot(args.root, args.techniques, args.test_type, args.save_dir, args.colours)
    #
    # # example running:
    #
    # # python plotting.py --plot_func performance --root ./output/ --test_type sensitivity --plot_type max --save_dir ./output/plots/