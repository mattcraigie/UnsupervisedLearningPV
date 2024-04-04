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

def null_test(model, num_patches, hist_save_path=None, results_save_path=None):

    # Create the null dataset
    null_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 0.5, 4, 8)
    null_dataset = TensorDataset(torch.from_numpy(null_data))
    null_dataloader = DataLoader(null_dataset, batch_size=64, shuffle=False)

    # Create the parity violating dataset
    parity_violating_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 1, 4, 8)
    parity_violating_dataset = TensorDataset(torch.from_numpy(parity_violating_data))
    parity_violating_dataloader = DataLoader(parity_violating_dataset, batch_size=64, shuffle=False)

    # Get the bootstrapped means
    null_means = get_bootstrap_means(model, null_dataloader, num_bootstraps=10000)

    # check that the null means are within 3 sigma of zero
    print("$\\mu_0^*$ mean: ", null_means.mean())
    print("$\\mu_0^*$ std: ", null_means.std())
    print("Standard deviations from zero: ", null_means.mean().abs() / null_means.std())
    print("")

    parity_violating_means = get_bootstrap_means(model, parity_violating_dataloader, num_bootstraps=10000)
    print("$\\mu^*$ mean: ", parity_violating_means.mean())
    print("$\\mu^*$ std: ", parity_violating_means.std())
    print("Standard deviations from zero: ", parity_violating_means.mean().abs() / parity_violating_means.std())
    print("")

    # check that these means come from the same distribution with a K-S test
    shifted_null_means = null_means - null_means.mean()
    shifted_parity_violating_means = parity_violating_means - parity_violating_means.mean()
    ks_stat, p_val = stats.ks_2samp(shifted_null_means, shifted_parity_violating_means)
    print("K-S test statistic: ", ks_stat)
    print("K-S test p-value: ", p_val)
    print("")

    if hist_save_path is not None:
        # plot the two distributions
        fig = plt.figure()
        plt.hist(shifted_null_means, bins=100, alpha=0.5, label="$\\mu_0^*$")
        plt.hist(shifted_parity_violating_means, bins=100, alpha=0.5, label="$\\mu^*$")
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel("Means after Bootstrapping")
        plt.ylabel("Frequency")
        plt.title("Null vs. Parity Violating Means")

        plt.savefig(hist_save_path)

    if results_save_path is not None:
        with open(results_save_path, 'w') as f:
            f.write("Null Means\n")
            f.write("Mean: {}\n".format(null_means.mean()))
            f.write("Standard Deviation: {}\n".format(null_means.std()))
            f.write("Standard Deviations from Zero: {}\n".format(null_means.mean().abs() / null_means.std()))
            f.write("\n")
            f.write("Parity Violating Means\n")
            f.write("Mean: {}\n".format(parity_violating_means.mean()))
            f.write("Standard Deviation: {}\n".format(parity_violating_means.std()))
            f.write("Standard Deviations from Zero: {}\n".format(parity_violating_means.mean().abs() / parity_violating_means.std()))
            f.write("\n")
            f.write("K-S Test\n")
            f.write("K-S Test Statistic: {}\n".format(ks_stat))
            f.write("K-S Test P-Value: {}\n".format(p_val))


def cosmic_variance_test(model, num_patches, num_universes, hist_save_path=None, results_save_path=None):

    # create the bootstrapped distribution
    bootstrap_data = create_parity_violating_mocks_2d(num_universes, 32, 16, 1, 4, 8)
    bootstrap_dataset = TensorDataset(torch.from_numpy(bootstrap_data))
    bootstrap_dataloader = DataLoader(bootstrap_dataset, batch_size=64, shuffle=False)

    bootstrap_means = get_bootstrap_means(model, bootstrap_dataloader, num_bootstraps=num_universes)

    # create the cosmic variance distribution
    all_universe_means = []
    for i in range(num_universes):
        universe_data = create_parity_violating_mocks_2d(num_patches, 32, 16, 1, 4, 8)
        universe_dataset = TensorDataset(torch.from_numpy(universe_data))
        universe_dataloader = DataLoader(universe_dataset, batch_size=64, shuffle=False)

        universe_mean = get_diffs(model, universe_dataloader)
        all_universe_means.append(universe_mean)

    all_universe_means = torch.cat(all_universe_means)

    print("Bootstrap mean: ", bootstrap_means.mean())
    print("Bootstrap std: ", bootstrap_means.std())

    print("Cosmic variance mean: ", all_universe_means.mean())
    print("Cosmic variance std: ", all_universe_means.std())

    # check that these means come from the same distribution with a K-S test
    ks_stat, p_val = stats.ks_2samp(bootstrap_means, all_universe_means)
    print("K-S test statistic: ", ks_stat)
    print("K-S test p-value: ", p_val)
    print("")

    if hist_save_path is not None:
        # plot the two distributions, only showing the top of the histograms
        fig = plt.figure()
        plt.hist(bootstrap_means, bins=100, alpha=0.5, label="\mu^*")
        plt.hist(all_universe_means, bins=100, alpha=0.5, label="$\\mu^\\star$")
        plt.xlabel("Means after Bootstrapping")
        plt.ylabel("Frequency")
        plt.title("Bootstrap vs. Cosmic Variance Means")
        plt.legend()

        plt.savefig(hist_save_path)

    if results_save_path is not None:
        with open(results_save_path, 'w') as f:
            f.write("Bootstrap Means\n")
            f.write("Mean: {}\n".format(bootstrap_means.mean()))
            f.write("Standard Deviation: {}\n".format(bootstrap_means.std()))
            f.write("\n")
            f.write("Cosmic Variance Means\n")
            f.write("Mean: {}\n".format(all_universe_means.mean()))
            f.write("Standard Deviation: {}\n".format(all_universe_means.std()))
            f.write("\n")
            f.write("K-S Test\n")
            f.write("K-S Test Statistic: {}\n".format(ks_stat))
            f.write("K-S Test P-Value: {}\n".format(p_val))


model_lookup = {'nfst': NFSTRegressor, 'mst': MSTRegressor, 'cnn': CNN}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # redo the above with shortened options
    parser.add_argument('-m', '--model_save_path', type=str, required=True)
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-s', '--save_dir', type=str, required=True)
    parser.add_argument('-n', '--num_neurons', type=int, default=128, required=False)
    parser.add_argument('-p', '--num_patches', type=int, default=1000, required=False)
    parser.add_argument('-u', '--num_universes', type=int, default=1000, required=False)

    args = parser.parse_args()

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

    # Run the null test
    null_test(model, args.num_patches,
              hist_save_path=os.path.join(args.save_dir, "null_test.png"),
              results_save_path=os.path.join(args.save_dir, "null_test_results.txt"))

    # Run the cosmic variance test
    cosmic_variance_test(model, args.num_patches, args.num_universes,
                         hist_save_path=os.path.join(args.save_dir, "cosmic_variance_test.png"),
                         results_save_path=os.path.join(args.save_dir, "cosmic_variance_test_results.txt"))

