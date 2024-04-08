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


def plot_histograms(save_dir):

    # NULL TEST

    # load the means
    null_means = np.load(os.path.join(save_dir, 'null_means.npy'))
    parity_violating_means = np.load(os.path.join(save_dir, 'parity_violating_means.npy'))

    # plot the two distributions
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    axes[0].hist(parity_violating_means, bins=50, alpha=0.6, label="PV Bootstrapped Means ($\\mu^*$)")
    counts, bins = np.histogram(null_means, bins=50)
    axes[0].stairs(counts, bins, label='Non-PV Bootstrapped Means ($\\mu_0^*$)', linewidth=2, ec='black')

    # axes[0].hist(a, bins=50, alpha=0.5, label="$\\mu_0^*$", color='black')
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_xlabel("Means")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # plot the shifted distributions
    shifted_null_means = null_means - null_means.mean()
    shifted_parity_violating_means = parity_violating_means - parity_violating_means.mean()

    axes[1].hist(shifted_parity_violating_means, bins=50, alpha=0.6, label="Centred $\\mu^*$")
    counts, bins = np.histogram(shifted_null_means, bins=50)
    axes[1].stairs(counts, bins, label='Centred $\\mu_0^*$', linewidth=2, ec='black')
    axes[1].legend()
    axes[1].set_xlabel("Shifted Means")

    plt.suptitle("Parity Violation Detection Model Verification")
    plt.savefig(os.path.join(save_dir, 'null_histograms.png'))


    # COSMIC VARIANCE TEST

    # load the means
    bootstrap_means = np.load(os.path.join(save_dir, 'parity_violating_means.npy')) #np.load(os.path.join(save_dir, 'bootstrap_means.npy'))
    all_universe_means = np.load(os.path.join(save_dir, 'all_universe_means.npy'))

    # plot the two distributions
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    axes[0].hist(bootstrap_means, bins=50, alpha=0.6, label="Single Survey Bootstrapped Means ($\\mu^*$)")
    counts, bins = np.histogram(all_universe_means, bins=50)
    axes[0].stairs(counts, bins, label='Cosmic Variance Means ($\\mu^\\star$)', linewidth=2, ec='darkgreen')

    # axes[0].hist(a, bins=50, alpha=0.5, label="$\\mu_0^*$", color='black')
    # axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_xlabel("Means")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # plot the shifted distributions
    shifted_bootstrap_means = bootstrap_means - bootstrap_means.mean()
    shifted_all_universe_means = all_universe_means - all_universe_means.mean()

    axes[1].hist(shifted_bootstrap_means, bins=50, alpha=0.6, label="Centred $\\mu^*$")
    counts, bins = np.histogram(shifted_all_universe_means, bins=50)
    axes[1].stairs(counts, bins, label='Centred $\\mu^\\star$', linewidth=2, ec='darkgreen')
    axes[1].legend()
    axes[1].set_xlabel("Shifted Means")

    plt.suptitle("Cosmic Variance Verification")
    plt.savefig(os.path.join(save_dir, 'cosmic_variance_histograms.png'))


    # NULL COSMIC VARIANCE TEST

    # load the means
    bootstrap_means = np.load(os.path.join(save_dir, 'parity_violating_means.npy')) #np.load(os.path.join(save_dir, 'bootstrap_means.npy'))
    null_all_universe_means = np.load(os.path.join(save_dir, 'null_all_universe_means.npy'))

    # plot the two distributions
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    axes[0].hist(bootstrap_means, bins=50, alpha=0.6, label="Single Survey Bootstrapped Means ($\\mu^*$)")
    counts, bins = np.histogram(null_all_universe_means, bins=50)
    axes[0].stairs(counts, bins, label='Null Cosmic Variance Means ($\\mu_0^\\star$)', linewidth=2, ec='maroon')

    # axes[0].hist(a, bins=50, alpha=0.5, label="$\\mu_0^*$", color='black')
    # axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_xlabel("Means")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # plot the shifted distributions
    shifted_bootstrap_means = bootstrap_means - bootstrap_means.mean()
    shifted_null_all_universe_means = null_all_universe_means - null_all_universe_means.mean()

    axes[1].hist(shifted_bootstrap_means, bins=50, alpha=0.6, label="Centred $\\mu^*$")
    counts, bins = np.histogram(shifted_null_all_universe_means, bins=50)
    axes[1].stairs(counts, bins, label='Centred $\\mu_0^\\star$', linewidth=2, ec='maroon')
    axes[1].legend()
    axes[1].set_xlabel("Shifted Means")

    plt.suptitle("Null Cosmic Variance Verification")
    plt.savefig(os.path.join(save_dir, 'null_cosmic_variance_histograms.png'))


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
