import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scattering_transform.filters import ClippedMorlet
from models import NFSTRegressor
import yaml
import argparse


def plot_filter_transformations(model, save_dir, transform_fn, title, file_name):
    """
    Plot filter transformations.

    Parameters:
    - model: The model containing the filters.
    - save_dir: Directory to save the plots.
    - transform_fn: Function to transform the filters.
    - title: Title for the plots.
    - file_name: File name to save the plot.
    """
    num_scales = model.filters.num_scales
    filters_final = [filt.clone() for filt in model.filters.filters]
    model.filters.load_state_dict(model.initial_filters_state)
    model.filters.update_filters()
    filters_initial = model.filters.filters

    fig, axes = plt.subplots(nrows=3, ncols=num_scales, figsize=(9, 9), dpi=100)

    for j in range(num_scales):
        filt_final = transform_fn(filters_final[j][0].cpu().detach())
        filt_initial = transform_fn(filters_initial[j][0].cpu().detach())

        filt_difference = filt_final - filt_initial
        norm_difference = TwoSlopeNorm(vmin=filt_difference.min(), vcenter=0, vmax=filt_difference.max())

        filt_asymmetry = filt_final.clone()
        filt_asymmetry[1:] = filt_asymmetry[1:] - filt_asymmetry[1:].flip(0)
        filt_asymmetry[0, :] = 0

        axes[0, j].imshow(filt_final)
        axes[1, j].imshow(filt_difference, norm=norm_difference, cmap='coolwarm')
        axes[2, j].imshow(filt_asymmetry, cmap='coolwarm')

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


def compare_filters(model, save_dir):
    # Fourier space transformation (no change)
    def fourier_real_transform(filter_tensor):
        return torch.fft.fftshift(filter_tensor)

    # Configuration space magnitude (no change)
    def config_abs_transform(filter_tensor):
        return torch.fft.fftshift(torch.fft.fft2(filter_tensor).abs())

    # Configuration space real part
    def config_real_transform(filter_tensor):
        return torch.fft.fftshift(torch.fft.fft2(filter_tensor).real)

    # Configuration space imaginary part
    def config_imaginary_transform(filter_tensor):
        return torch.fft.fftshift(torch.fft.fft2(filter_tensor).imag)

    # Plot in Fourier space
    plot_filter_transformations(model, save_dir, fourier_real_transform,
                                'NFST Learned Filters: Fourier Space Real Part',
                                'filters_k.png')

    # Plot in Configuration space (Magnitude)
    plot_filter_transformations(model, save_dir, config_abs_transform,
                                'NFST Learned Filters: Configuration Space Magnitudes',
                                'filters_x_abs.png')

    # Plot in Configuration space (Real Part)
    plot_filter_transformations(model, save_dir, config_real_transform,
                                'NFST Learned Filters: Configuration Space Real Part',
                                'filters_x_real.png')

    # Plot in Configuration space (Imaginary Part)
    plot_filter_transformations(model, save_dir, config_imaginary_transform,
                                'NFST Learned Filters: Configuration Space Imaginary Part',
                                'filters_x_imag.png')


def show_filters(model, save_dir, morlet_diff=False):

    num_scales = model.filters.num_scales
    num_angles = model.filters.num_angles

    morlet_reference = ClippedMorlet(32, num_scales, num_angles)
    morlet_reference.clip_filters()

    fig, axes = plt.subplots(nrows=num_scales, ncols=4, figsize=(12, 9), dpi=100)
    filts = model.filters.filters

    for j in range(num_scales):

        filt_k = filts[j][0]
        filt_x = torch.fft.fft2(filt_k)

        filt_x = torch.fft.fftshift(filt_x).detach()
        filt_k = torch.fft.fftshift(filt_k).detach()

        if not morlet_diff:
            axes[j, 0].imshow(filt_k)
            axes[j, 1].imshow(filt_x.real)
            axes[j, 2].imshow(filt_x.imag)
            axes[j, 3].imshow(filt_x.abs())

        else:
            base_k = morlet_reference.filters[j][3]
            base_x = torch.fft.fft2(base_k)
            base_x = torch.fft.fftshift(base_x).detach()
            base_k = torch.fft.fftshift(base_k).detach()

            axes[j, 0].imshow(filt_k - base_k)
            axes[j, 1].imshow(filt_x.real - base_x.real)
            axes[j, 2].imshow(filt_x.imag - base_x.imag)
            axes[j, 3].imshow(filt_x.abs() - base_x.abs())

        for a in range(4):
            axes[j, a].set_axis_off()

    plt.savefig(os.path.join(save_dir, 'filters.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # redo the above with shortened options
    parser.add_argument('-m', '--model_save_path', type=str, required=True)
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-s', '--save_dir', type=str, required=True)


    args = parser.parse_args()

    # load config yaml
    config_path = os.path.join(args.config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = NFSTRegressor(**config['analysis_kwargs']['model_kwargs'])
    model.load_state_dict(torch.load(args.model_save_path))

    # show filters
    compare_filters(model, args.save_dir)


