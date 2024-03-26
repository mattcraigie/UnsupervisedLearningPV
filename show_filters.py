import os
import torch
import matplotlib.pyplot as plt
from scattering_transform.filters import ClippedMorlet
from models import NFSTRegressor
import yaml
import argparse

from torch.fft import fftshift as fts


def compare_filters(model, save_dir):

    num_scales = model.filters.num_scales
    filters_final = model.filters.filters
    filters_final = [filt.clone() for filt in filters_final]

    model.filters.load_state_dict(model.initial_filters_state)
    model.filters.update_filters()
    filters_initial = model.filters.filters

    # figure 1 - Fourier space
    fig, axes = plt.subplots(nrows=3, ncols=num_scales, figsize=(9, 9), dpi=100)

    for j in range(num_scales):
        # plot the final filters, and then the difference between the final and initial filters
        filt_k0 = filters_final[j][0].cpu().detach()
        filt_k1 = filters_initial[j][0].cpu().detach()

        filt_k1_symm = filt_k1.clone()
        filt_k1_symm[1:] = filt_k1_symm[1:] - filt_k1_symm[1:].flip(0)

        axes[0, j].imshow(fts(filt_k1))
        axes[1, j].imshow(fts(filt_k1 - filt_k0))
        axes[2, j].imshow(fts(filt_k1_symm))

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_title('$j=0$')
    axes[0, 1].set_title('$j=1$')
    axes[0, 2].set_title('$j=2$')

    axes[0, 0].set_ylabel('Learned Filter', fontsize=14)
    axes[1, 0].set_ylabel('Difference from \nMorlet Initialisation', fontsize=14)
    axes[2, 0].set_ylabel('Asymmetry in Filter', fontsize=14)

    plt.suptitle('NFST Learned Filters: Fourier Space Magntiudes')
    plt.savefig(os.path.join(save_dir, 'filters_k.png'))

    # figure 2 - Configuration space
    fig, axes = plt.subplots(nrows=3, ncols=num_scales, figsize=(9, 9), dpi=100)

    for j in range(num_scales):
        # plot the final filters, and then the difference between the final and initial filters
        filt_k0 = filters_final[j][0].cpu().detach()
        filt_x0 = fts(torch.fft.fft2(filt_k0).abs())
        filt_k1 = filters_initial[j][0].cpu().detach()
        filt_x1 = fts(torch.fft.fft2(filt_k1).abs())

        filt_k1_symm = filt_k1.clone()
        filt_k1_symm[1:] = filt_k1_symm[1:] - filt_k1_symm[1:].flip(0)

        filt_x1_symm = filt_x1.clone()
        filt_x1_symm[1:] = filt_x1_symm[1:] - filt_x1_symm[1:].flip(0)

        axes[0, j].imshow(filt_x1)
        axes[1, j].imshow(filt_x1 - filt_x0)
        axes[2, j].imshow(filt_x1_symm)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_title('$j=0$')
    axes[0, 1].set_title('$j=1$')
    axes[0, 2].set_title('$j=2$')

    axes[0, 0].set_ylabel('Learned Filter', fontsize=14)
    axes[1, 0].set_ylabel('Difference from \nMorlet Initialisation', fontsize=14)
    axes[2, 0].set_ylabel('Asymmetry in Filter', fontsize=14)

    plt.suptitle('NFST Learned Filters: Configuration Space Magntiudes')

    plt.savefig(os.path.join(save_dir, 'filters_x.png'))


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


