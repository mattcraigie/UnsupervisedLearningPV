from dlutils.training import RegressionTrainer
from dlutils.data import DataHandler, RotatedDataset
import torch
import numpy as np
import os
from scipy.stats import ks_2samp
import time

from models import NFSTRegressor, CNN, MSTRegressor
from bootstrapping import get_bootstrap_score, get_bootstrap_means, get_diffs
from mocks import create_parity_violating_mocks_2d

model_lookup = {'nfst': NFSTRegressor, 'mst': MSTRegressor, 'cnn': CNN}


def batch_difference_loss(model, data):
    gx = model(data)  # g(x), the model evaluated on the field
    gPx = model(data.flip(dims=(-1,)))  # g(Px), the model evaluated on the field with a parity operation applied (flip)

    fx = gx - gPx  # f(x), the parity violating statistic

    mu_B = fx.mean(0)
    sigma_B = fx.std(0)

    return -mu_B / sigma_B


def train_and_test_model(model_type, model_name, model_kwargs, mock_kwargs, training_kwargs, output_root, repeats=1,
                         num_verification_catalogs=None, device=None):

    try:
        model_class = model_lookup[model_type]
    except KeyError:
        raise KeyError(f'Unrecognized model type {model_type} for {model_name} analysis')

    # make the save dirs for each repeat
    save_dirs = []
    for repeat in range(repeats):
        save_dirs.append(os.path.join(output_root, f'repeat_{repeat}'))
        os.makedirs(save_dirs[-1], exist_ok=True)

    ### ~~~ TRAINING ~~~ ###

    validation_scores = []
    test_scores = []

    test_mocks = create_parity_violating_mocks_2d(training_kwargs['num_test_mocks'], **mock_kwargs)
    test_mocks = torch.from_numpy(test_mocks).float().unsqueeze(1)

    test_data_handler = DataHandler(test_mocks)

    test_loader = test_data_handler.make_single_dataloader(batch_size=64)

    for repeat in range(repeats):
        start = time.time()

        np.random.seed(0)  # mocks are always the same
        torch.manual_seed(repeat)  # model is different each time, but consistent for the same repeat

        train_val_mocks = create_parity_violating_mocks_2d(training_kwargs['num_train_val_mocks'], **mock_kwargs)
        train_val_mocks = torch.from_numpy(train_val_mocks).float().unsqueeze(1)

        # we keep a consistent train and validation set across all repeats
        data_handler = DataHandler(train_val_mocks)
        train_loader, val_loader = data_handler.make_dataloaders(batch_size=64, shuffle_split=False,
                                                                 shuffle_dataloaders=True, val_fraction=0.2,
                                                                 dataset_class=RotatedDataset)

        model = model_class(**model_kwargs)
        model.to(device)

        trainer = RegressionTrainer(model, train_loader, val_loader, criterion=batch_difference_loss, no_targets=True, device=device)
        trainer.run_training(epochs=training_kwargs['epochs'], lr=training_kwargs['lr'], print_progress=False, show_loss_plot=False)

        # save the best model
        trainer.get_best_model()
        torch.save(model.state_dict(), os.path.join(save_dirs[repeat], 'model.pt'))

        # save the loss curves
        trainer.save_losses(os.path.join(save_dirs[repeat], 'losses.npy'))

        # get the validation booststrap score
        val_bootstrap_score = get_bootstrap_score(model, val_loader)
        validation_scores.append(val_bootstrap_score)

        test_bootstrap_score = get_bootstrap_score(model, test_loader)
        test_scores.append(test_bootstrap_score)

        print(f'Repeat {repeat} took {time.time() - start:.2f} seconds')

    validation_scores = np.array(validation_scores)
    test_scores = np.array(test_scores)

    output_dict = {'val_scores': validation_scores, 'test_scores': test_scores}

    ### ~~~ VERIFICATION ~~~ ###

    # if num_verification_catalogs is not None:
    #     best_model = model_class(**model_kwargs)
    #     best_model.load_state_dict(torch.load(os.path.join(save_dirs[np.argmax(test_scores)], 'model.pt')))
    #     best_model.to(device)
    #
    #     # run the best model on the full many-universe test to verify the bootstrap std matches the std of the many-universe test
    #
    #     bootstrap_means = get_bootstrap_means(best_model, test_loader)
    #     bootstrap_std = bootstrap_means.std()
    #     output_dict['bootstrap_std'] = bootstrap_std.item()
    #
    #
    #     verification_means = []
    #     for i in range(num_verification_catalogs):
    #         verification_mocks = create_parity_violating_mocks_2d(training_kwargs['num_test_mocks'], **mock_kwargs)
    #         verification_mocks = torch.from_numpy(verification_mocks).float().unsqueeze(1)
    #         data_handler = DataHandler(verification_mocks)
    #         verification_loader = data_handler.make_single_dataloader(batch_size=64)
    #         verification_means.append(get_diffs(best_model, verification_loader).mean())
    #
    #     verification_means = torch.tensor(verification_means)
    #     verification_std = verification_means.std()
    #
    #     output_dict['verification_std'] = verification_std.item()
    #
    #     # 2 sample KS test
    #     bootstrap_means = bootstrap_means.numpy().squeeze(1)
    #     verification_means = verification_means.numpy()
    #
    #     shifted_bootstrap_means = bootstrap_means - bootstrap_means.mean()
    #     shifted_verification_means = verification_means - verification_means.mean()
    #
    #     output_dict['ks_test_pvalue'] = ks_2samp(shifted_bootstrap_means, shifted_verification_means).pvalue

    return output_dict

