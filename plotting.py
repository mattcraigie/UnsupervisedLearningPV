import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def losses_plot(root, techniques, test_type, save_dir):
    if test_type == 'sensitivity':
        file_start = 'ratio'
    elif test_type == 'data_scaling':
        file_start = 'datascaling'

    for technique in techniques:
        technique_dir = os.path.join(root, test_type, technique)

        sub_folders = []
        for filename in os.listdir(technique_dir):
            full_path = os.path.join(technique_dir, filename)  # Concatenating the directory and filename
            if os.path.isdir(full_path) and filename.startswith(file_start):
                sub_folders.append(full_path)

        sub_folders = np.sort(sub_folders)

        fig, axes = plt.subplots(ncols=len(sub_folders), nrows=2, figsize=(len(sub_folders) * 3, 4), sharey=False,
                                 dpi=300)
        plt.suptitle(technique + ' losses')

        for i, sub_folder in enumerate(sub_folders):

            for repeat_folder in os.listdir(sub_folder):
                # need to decode utf-8 because this is reading in as byte strings??? not sure why
                loss_path = os.path.join(sub_folder, repeat_folder.decode("utf-8"), 'losses.npy')
                losses = np.load(loss_path)
                axes[0, i].plot(losses[0], c='blue')
                axes[1, i].plot(losses[1], c='red')
                # axes[i].set_xticks([])

                # break
            axes[0, i].set_title(sub_folder.split('/')[-1])
            axes[0, i].set_yticks([])
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel('Train Loss')
        axes[1, 0].set_ylabel('Val Loss')

        plt.savefig(os.path.join(save_dir, test_type + '_' + technique + '_losses.png'))


def performance_plot(root, techniques, test_type, plot_type, save_dir, colours=None):
    assert plot_type in ['max', 'mean', 'all'], 'type must be one of "max", "mean", or "all"'
    assert test_type in ['sensitivity', 'data_scaling'], 'test_type must be one of "sensitivity" or "datascaling"'

    key = 'ratio_left' if test_type == 'sensitivity' else 'num_train_val_mocks'
    csv_name = 'sensitivity.csv' if test_type == 'sensitivity' else 'data_scaling.csv'

    if colours is None:
        colours = ['red', 'goldenrod', 'blue', 'pink']

    for colour, technique in zip(colours, techniques):
        df = pd.read_csv(os.path.join(root, test_type, technique, csv_name))
        ratios = df[key]




        scores = np.load(os.path.join(root, test_type, technique, 'training_scores.npy'))

        if plot_type == 'max':
            plt.scatter(ratios, scores.max(1), c=colour, label=technique)
        elif plot_type == 'mean':
            plt.scatter(ratios, scores.mean(1), c=colour, label=technique)
        elif plot_type == 'all':
            for i in range(scores.shape[1]):
                if i == 0:
                    plt.scatter(ratios, scores[:, i], c=colour, label=technique)
                else:
                    plt.scatter(ratios, scores[:, i], c=colour)

        plt.axhline(3, c='black', linestyle=':')
    plt.legend()

    if test_type == 'sensitivity':
        plt.xlabel('Ratio of Training to Test Data')
    elif test_type == 'data_scaling':
        plt.xlabel('Number of Samples in Train/Val Set')
        plt.semilogx()
    plt.ylabel('Test Score ($\eta$)')

    plt.savefig(os.path.join(save_dir, plot_type + '_' + test_type + '_performance.png'))


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

    plt.savefig(os.path.join(save_dir, 'verification.png'))


if __name__ == '__main__':
    # parse arguments for plotting
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_func', type=str, default='performance')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--techniques', type=str, nargs='+', default=['nfst', 'nfst_mini', 'mst', 'cnn'])
    parser.add_argument('--test_type', type=str, default='sensitivity')
    parser.add_argument('--plot_type', type=str, default='max')
    parser.add_argument('--save_dir', type=str, default='./output/plots/')
    parser.add_argument('--colours', type=str, nargs='+', default=None)

    # also for the type of plotting function to call: losses, performance, verification


    args = parser.parse_args()

    if args.plot_func == 'losses':
        losses_plot(args.root, args.techniques, args.test_type, args.save_dir)

    elif args.plot_func == 'performance':
        performance_plot(args.root, args.techniques, args.test_type, args.plot_type, args.save_dir, args.colours)

    elif args.plot_func == 'verification':
        verification_plot(args.root, args.techniques, args.test_type, args.save_dir, args.colours)

    # example running:

    # python plotting.py --plot_func performance --root ./output/ --test_type sensitivity --plot_type max --save_dir ./output/plots/