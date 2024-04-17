"""
This script contains three tests.

The first test verifies that the bootstrapped means of the model applied to the parity violating data (i.e. the data
with signal) is equivalent to the bootstrapped means of the null dataset

The second test verifies that the bootstrapped means are equivalent to the cosmic variance, i.e. a full, independent
dataset with the same signal.

The third test verifies that everything is Gaussian, because this is an underlying assumption.

"""
from mocks import *
from bootstrapping import get_bootstrap_means, get_diffs
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from models import NFSTRegressor, CNN, MSTRegressor
import torch
import numpy as np


model_lookup = {'nfst': NFSTRegressor, 'mst': MSTRegressor, 'cnn': CNN}


def null_test(model, num_patches, save_dir=None):

    # Create the null dataset
    null_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 0.5, 4, 8)
    null_dataset = TensorDataset(torch.from_numpy(null_data).unsqueeze(1).float())
    null_dataloader = DataLoader(null_dataset, batch_size=64, shuffle=False)

    # Create the parity violating dataset
    parity_violating_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 1, 4, 8)
    parity_violating_dataset = TensorDataset(torch.from_numpy(parity_violating_data).unsqueeze(1).float())
    parity_violating_dataloader = DataLoader(parity_violating_dataset, batch_size=64, shuffle=False)

    # Get the bootstrapped means and see the standard deviations from zero
    null_means = get_bootstrap_means(model, null_dataloader, num_bootstraps=10000)
    null_std_devs_from_zero = null_means.mean().abs() / null_means.std()

    parity_violating_means = get_bootstrap_means(model, parity_violating_dataloader, num_bootstraps=10000)
    pv_std_devs_from_zero = parity_violating_means.mean().abs() / parity_violating_means.std()

    # check that these means come from the same distribution with a K-S test
    shifted_null_means = (null_means - null_means.mean()).squeeze(1).numpy()
    shifted_parity_violating_means = (parity_violating_means - parity_violating_means.mean()).squeeze(1).numpy()
    ks_stat, p_val = stats.ks_2samp(shifted_null_means, shifted_parity_violating_means)

    # write the above to a file, maintaining the same formatting
    save_file = os.path.join(save_dir, 'null_test_results.txt')

    with open(save_file, 'a') as f:
        f.write("\n\n\n~~~ NULL TEST ~~~\n\n")
        f.write("$\\mu_0^*$ mean: {:.3e}\n".format(null_means.mean().item()))
        f.write("$\\mu_0^*$ std: {:.3e}\n".format(null_means.std().item()))
        f.write("Standard deviations from zero: {:.3e}\n\n".format(null_std_devs_from_zero.item()))
        f.write("$\\mu^*$ mean: {:.3e}\n".format(parity_violating_means.mean().item()))
        f.write("$\\mu^*$ std: {:.3e}\n".format(parity_violating_means.std().item()))
        f.write("Standard deviations from zero: {:.3e}\n\n".format(pv_std_devs_from_zero.item()))
        f.write("K-S test statistic: {:.3e}\n".format(ks_stat))
        f.write("K-S test p-value: {:.3e}\n\n".format(p_val))

    # save the bootstrap and cosmic variance means
    np.save(os.path.join(save_dir, 'null_means.npy'), null_means.squeeze(1).numpy())
    np.save(os.path.join(save_dir, 'parity_violating_means.npy'), parity_violating_means.squeeze(1).numpy())


def cosmic_variance_test(model, num_patches, num_universes, save_dir=None):

    # create the bootstrapped distribution
    bootstrap_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 1, 4, 8)
    bootstrap_dataset = TensorDataset(torch.from_numpy(bootstrap_data).unsqueeze(1).float())
    bootstrap_dataloader = DataLoader(bootstrap_dataset, batch_size=64, shuffle=False)

    bootstrap_means = get_bootstrap_means(model, bootstrap_dataloader, num_bootstraps=10000)

    # create the cosmic variance distribution
    all_universe_means = []
    for i in range(num_universes):
        universe_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 1, 4, 8)
        universe_dataset = TensorDataset(torch.from_numpy(universe_data).unsqueeze(1).float())
        universe_dataloader = DataLoader(universe_dataset, batch_size=64, shuffle=False)

        universe_mean = get_diffs(model, universe_dataloader).mean()
        all_universe_means.append(universe_mean)

    all_universe_means = torch.stack(all_universe_means)

    # check that these means come from the same distribution with a K-S test
    shifted_bootstrap_means = (bootstrap_means - bootstrap_means.mean()).squeeze(1).numpy()
    shifted_all_universe_means = (all_universe_means - all_universe_means.mean()).numpy()
    ks_stat, p_val = stats.ks_2samp(shifted_bootstrap_means, shifted_all_universe_means)


    save_file = os.path.join(save_dir, 'cosmic_variance_test_results.txt')
    with open(save_file, 'a') as f:
        f.write("\n\n\n~~~ COSMIC VARIANCE TEST ~~~\n\n")
        f.write("Bootstrap mean: {:.3e}\n".format(bootstrap_means.mean().item()))
        f.write("Bootstrap std: {:.3e}\n\n".format(bootstrap_means.std().item()))
        f.write("Cosmic variance mean: {:.3e}\n".format(all_universe_means.mean().item()))
        f.write("Cosmic variance std: {:.3e}\n\n".format(all_universe_means.std().item()))
        f.write("K-S test statistic: {:.3e}\n".format(ks_stat))
        f.write("K-S test p-value: {:.3e}\n\n".format(p_val))

    # save the bootstrap and cosmic variance means
    np.save(os.path.join(save_dir, 'bootstrap_means.npy'), bootstrap_means.squeeze(1).numpy())
    np.save(os.path.join(save_dir, 'all_universe_means.npy'), all_universe_means.numpy())

def null_cosmic_variance_test(model, num_patches, num_universes, save_dir=None):

    # create the bootstrapped distribution
    bootstrap_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 1, 4, 8)
    bootstrap_dataset = TensorDataset(torch.from_numpy(bootstrap_data).unsqueeze(1).float())
    bootstrap_dataloader = DataLoader(bootstrap_dataset, batch_size=64, shuffle=False)

    bootstrap_means = get_bootstrap_means(model, bootstrap_dataloader, num_bootstraps=10000)

    # create the cosmic variance distribution
    all_universe_means = []
    for i in range(num_universes):
        universe_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 0.5, 4, 8)
        universe_dataset = TensorDataset(torch.from_numpy(universe_data).unsqueeze(1).float())
        universe_dataloader = DataLoader(universe_dataset, batch_size=64, shuffle=False)

        universe_mean = get_diffs(model, universe_dataloader).mean()
        all_universe_means.append(universe_mean)

    all_universe_means = torch.stack(all_universe_means)

    # check that these means come from the same distribution with a K-S test
    shifted_bootstrap_means = (bootstrap_means - bootstrap_means.mean()).squeeze(1).numpy()
    shifted_all_universe_means = (all_universe_means - all_universe_means.mean()).numpy()
    ks_stat, p_val = stats.ks_2samp(shifted_bootstrap_means, shifted_all_universe_means)

    save_file = os.path.join(save_dir, 'null_cosmic_variance_test_results.txt')
    with open(save_file, 'a') as f:
        f.write("\n\n\n~~~ NULL COSMIC VARIANCE TEST ~~~\n\n")
        f.write("Bootstrap mean: {:.3e}\n".format(bootstrap_means.mean().item()))
        f.write("Bootstrap std: {:.3e}\n\n".format(bootstrap_means.std().item()))
        f.write("Cosmic variance mean: {:.3e}\n".format(all_universe_means.mean().item()))
        f.write("Cosmic variance std: {:.3e}\n\n".format(all_universe_means.std().item()))
        f.write("K-S test statistic: {:.3e}\n".format(ks_stat))
        f.write("K-S test p-value: {:.3e}\n\n".format(p_val))

    # save the bootstrap and cosmic variance means
    # np.save(os.path.join(save_dir, 'bootstrap_means.npy'), bootstrap_means.squeeze(1).numpy())
    np.save(os.path.join(save_dir, 'null_all_universe_means.npy'), all_universe_means.numpy())


    # NULL TEST

    # load the means
    # null_means = np.load(os.path.join(save_dir, 'null_means.npy'))
    # parity_violating_means = np.load(os.path.join(save_dir, 'parity_violating_means.npy'))
    #
    # # plot the two distributions
    # fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    # axes[0].hist(parity_violating_means, bins=50, alpha=0.6, label="PV Bootstrapped Means ($\\mu^*$)")
    # counts, bins = np.histogram(null_means, bins=50)
    # axes[0].stairs(counts, bins, label='Non-PV Bootstrapped Means ($\\mu_0^*$)', linewidth=2, ec='black')
    #
    # # axes[0].hist(a, bins=50, alpha=0.5, label="$\\mu_0^*$", color='black')
    # axes[0].axvline(0, color='black', linestyle='--')
    # axes[0].set_xlabel("Means")
    # axes[0].set_ylabel("Frequency")
    # axes[0].legend()
    #
    # # plot the shifted distributions
    # shifted_null_means = null_means - null_means.mean()
    # shifted_parity_violating_means = parity_violating_means - parity_violating_means.mean()
    #
    # axes[1].hist(shifted_parity_violating_means, bins=50, alpha=0.6, label="Centred $\\mu^*$")
    # counts, bins = np.histogram(shifted_null_means, bins=50)
    # axes[1].stairs(counts, bins, label='Centred $\\mu_0^*$', linewidth=2, ec='black')
    # axes[1].legend()
    # axes[1].set_xlabel("Shifted Means")
    #
    # plt.suptitle("Parity Violation Detection Model Verification")
    # plt.savefig(os.path.join(save_dir, 'null_histograms.png'))


def plot_two_histograms(ax, data_1, data_2, common_bins, labels, colors, legend=False):
    # data1 will be turned into a hist
    # data2 will be turned into a stairs plot

    bins_1 = common_bins + data_1.mean()
    bins_2 = common_bins + data_2.mean()

    # compute the hists
    hist_1, _ = np.histogram(data_1, bins=bins_1)
    hist_2, _ = np.histogram(data_2, bins=bins_2)

    # plot the histograms
    ax.hist(data_1, bins=bins_1, alpha=1, label=labels[0], color=colors[0])
    ax.stairs(hist_2, bins_2, label=labels[1], linewidth=2, ec=colors[1])

    if legend:
        # make a legend with no outline
        ax.legend(frameon=False)

    return ax


def make_histogram_plot(means_1, means_2, common_bins, labels, colors, save_path):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # normal plot
    plot_two_histograms(ax1, means_1, means_2, common_bins, labels, colors, legend=True)

    # shifted plot
    shifted_means_1 = means_1 - means_1.mean()
    shifted_means_2 = means_2 - means_2.mean()
    shifted_labels = [label + " (Centered)" for label in labels]
    plot_two_histograms(ax2, shifted_means_1, shifted_means_2, common_bins, shifted_labels, colors, legend=False)

    plt.savefig(save_path)


def generate_histograms(save_dir):

    # load the means
    bootstrap_means = np.load(os.path.join(save_dir, 'bootstrapped_means.npy'))
    pv_means = np.load(os.path.join(save_dir, 'all_universe_means.npy'))
    null_means = np.load(os.path.join(save_dir, 'null_all_universe_means.npy'))

    bootstrap_means_shifted = bootstrap_means - bootstrap_means.mean()
    pv_means_shifted = pv_means - pv_means.mean()
    null_means_shifted = null_means - null_means.mean()

    # establish common bins for all the shifted means, and make them symmetric
    num_bins = 50
    common_min = min(bootstrap_means_shifted.min(), pv_means_shifted.min(), null_means_shifted.min())
    common_max = max(bootstrap_means_shifted.max(), pv_means_shifted.max(), null_means_shifted.max())

    symm_min = -max(abs(common_min), abs(common_max))
    symm_max = max(abs(common_min), abs(common_max))

    common_bins = np.linspace(symm_min, symm_max, num_bins)

    # make the bootstrap/pv_universe plot
    labels = ['$\\mu^*$', '$\\mu^\\star$']
    colors = ['lightsteelblue', 'darkgreen']
    save_path = os.path.join(save_dir, 'pv_histograms.png')

    make_histogram_plot(bootstrap_means_shifted, pv_means_shifted, common_bins, labels, colors, save_path)

    # make the bootstrap/null_universe plot

    labels = ['$\\mu^*$', '$\\mu_0^\\star$']
    colors = ['lightsteelblue', 'black']
    save_path = os.path.join(save_dir, 'null_histograms.png')

    make_histogram_plot(bootstrap_means_shifted, null_means_shifted, common_bins, labels, colors, save_path)


def normality_test(bootstrap, pv_universe, null_universe):
    print("Normality Test")

    # normalise first
    bootstrap_norm = (bootstrap - bootstrap.mean()) / bootstrap.std()
    pv_universe_norm = (pv_universe - pv_universe.mean()) / pv_universe.std()
    null_universe_norm = (null_universe - null_universe.mean()) / null_universe.std()

    # normal test
    print("Bootstrap Means: {:.3e}".format(stats.normaltest(bootstrap_norm)[1]))
    print("PV CV Means: {:.3e}".format(stats.normaltest(pv_universe_norm)[1]))
    print("Null CV Means: {:.3e}".format(stats.normaltest(null_universe_norm)[1]))





if __name__ == '__main__':

    print("RUUNINGG???")

    parser = argparse.ArgumentParser()

    # redo the above with shortened options
    parser.add_argument('-m', '--model_save_path', type=str, required=True)
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-s', '--save_dir', type=str, required=True)
    parser.add_argument('-n', '--num_neurons', type=int, default=128, required=False)
    parser.add_argument('-p', '--num_patches', type=int, default=1000, required=False)
    parser.add_argument('-u', '--num_universes', type=int, default=1000, required=False)
    parser.add_argument('-a', '--already_run', type=bool, default=False, required=False)

    args = parser.parse_args()

    if not args.already_run:

        print('Running')

        # load config yaml
        config_path = os.path.join(args.config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if args.num_neurons is not None:
            config['analysis_kwargs']['model_kwargs']['subnet_hidden_sizes'] = [args.num_neurons, args.num_neurons]

        model_type = config['analysis_kwargs']['model_type']
        model_name = config['analysis_kwargs']['model_name']
        try:
            model_class = model_lookup[model_type]
        except KeyError:
            raise KeyError(f'Unrecognized model type {model_type} for {model_name} analysis')

        model = model_class(**config['analysis_kwargs']['model_kwargs'])
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(torch.device("cuda"))

        # Run the null bootstrap test
        # null_test(model, args.num_patches, args.save_dir)

        # Run the cosmic variance test
        # cosmic_variance_test(model, args.num_patches, args.num_universes, args.save_dir)

        # Run the null cosmic variance test
        null_cosmic_variance_test(model, args.num_patches, args.num_universes, args.save_dir)

    # plot the histograms
    plot_histograms(args.save_dir)
