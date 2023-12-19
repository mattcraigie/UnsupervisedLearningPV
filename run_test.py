import numpy as np
import os
import pandas as pd
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import argparse

from training_functions import train_and_test_model

from datetime import datetime


def pv_detection(config):
    """Run the data scaling analysis. Outputs the results to a csv file.

    Args:
        args (argparse.Namespace): The command line arguments. This function only uses args.config.
    """
    # Load configuration settings



    device = config['device']

    analysis_type = config['analysis_type']

    assert analysis_type in ['sensitivity', 'data_scaling'], f"Analysis type {analysis_type} not recognized."

    analysis_config = config['analysis_kwargs']

    output_root = analysis_config['output_root']

    analysis_folder = os.path.join(output_root, analysis_type)
    os.makedirs(analysis_folder, exist_ok=True)

    model_folder = os.path.join(analysis_folder, analysis_config['model_type'])
    os.makedirs(model_folder, exist_ok=True)

    # set up logging

    logging_filename = os.path.join(model_folder, f'{analysis_type}.log')


    if os.path.exists(logging_filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging_filename = os.path.join(model_folder, f'{analysis_type}_{timestamp}.log')

    logging.basicConfig(filename=logging_filename, level=logging.INFO)
    logging.info(f"Running on {device}.")


    start_time = time.time()

    logging.info("Running analysis.")

    if analysis_type == 'sensitivity':

        all_scores = []

        ratios = analysis_config['ratio_left']

        for i in range(len(ratios)):
            logging.info(f"Running with balance {ratios[i]}")
            analysis_config['mock_kwargs']['ratio_left'] = ratios[i]
            output_dict = train_and_test_model(**analysis_config, device=device)
            all_scores.append(output_dict)

        # save results to csv
        df = pd.DataFrame(all_scores)
        df.to_csv(os.path.join(model_folder, 'sensitivity.csv'), index=False)

    elif analysis_type == 'data_scaling':

        all_scores = []

        data_sizes = analysis_config['num_train_val_mocks']

        for i in range(len(data_sizes)):
            logging.info(f"Running with {data_sizes[i]} training and validation mocks")
            analysis_config['mock_kwargs']['num_train_val_mocks'] = data_sizes[i]
            output_dict = train_and_test_model(**analysis_config, device=device)
            all_scores.append(output_dict)

        # save results to csv
        df = pd.DataFrame(all_scores)
        df.to_csv(os.path.join(model_folder, 'data_scaling.csv'), index=False)



    end_time = time.time()
    logging.info("Analysis took {:.2f} seconds.".format(end_time - start_time))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the data scaling analysis.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # use cuda if args.cuda is set to True
    pv_detection(config)


