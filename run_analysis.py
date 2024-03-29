import os
import pandas as pd
import logging
import time
import numpy as np

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

    assert analysis_type in ['sensitivity', 'data_scaling', 'nfst_sizes'], f"Analysis type {analysis_type} not recognized."

    analysis_config = config['analysis_kwargs']

    output_root = analysis_config['output_root']

    analysis_folder = os.path.join(output_root, analysis_type)
    os.makedirs(analysis_folder, exist_ok=True)

    model_folder = os.path.join(analysis_folder, analysis_config['model_name'])
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

        all_training_scores = []
        all_outputs = []

        ratios = analysis_config['mock_kwargs']['ratio_left']

        for i in range(len(ratios)):

            ratio_folder = os.path.join(model_folder, f'ratio_{i}')
            os.makedirs(ratio_folder, exist_ok=True)
            analysis_config['output_root'] = ratio_folder

            analysis_config['mock_kwargs']['ratio_left'] = ratios[i]

            logging.info(f"Running with balance {ratios[i]}")

            training_scores, output_dict = train_and_test_model(**analysis_config, device=device)

            all_training_scores.append(training_scores)
            all_outputs.append(output_dict)

        # save results to csv
        df = pd.DataFrame(all_outputs)
        df['ratio_left'] = ratios
        df.to_csv(os.path.join(model_folder, 'sensitivity.csv'), index=False)

        np.save(os.path.join(model_folder, 'training_scores.npy'), np.stack(all_training_scores))

    elif analysis_type == 'data_scaling':

        all_training_scores = []
        all_outputs = []

        data_sizes = analysis_config['training_kwargs']['num_train_val_mocks']

        for i in range(len(data_sizes)):

            datascaling_folder = os.path.join(model_folder, f'datascaling_{i}')
            os.makedirs(datascaling_folder, exist_ok=True)
            analysis_config['output_root'] = datascaling_folder

            analysis_config['training_kwargs']['num_train_val_mocks'] = data_sizes[i]

            logging.info(f"Running with {data_sizes[i]} training and validation mocks")
            training_scores, output_dict = train_and_test_model(**analysis_config, device=device)

            all_training_scores.append(training_scores)
            all_outputs.append(output_dict)

        # save results to csv
        df = pd.DataFrame(all_outputs)
        df['num_train_val_mocks'] = data_sizes
        df.to_csv(os.path.join(model_folder, 'data_scaling.csv'), index=False)

        np.save(os.path.join(model_folder, 'training_scores.npy'), np.stack(all_training_scores))

    elif analysis_type == 'nfst_sizes':

        all_training_scores = []
        all_outputs = []

        nfst_sizes = analysis_config['model_kwargs']['subnet_hidden_sizes']

        for i in range(len(nfst_sizes)):

            nfst_sizes_folder = os.path.join(model_folder, f'nfst_sizes_{i}')
            os.makedirs(nfst_sizes_folder, exist_ok=True)
            analysis_config['output_root'] = nfst_sizes_folder

            analysis_config['model_kwargs']['subnet_hidden_sizes'] = nfst_sizes[i]

            logging.info(f"Running with {nfst_sizes[i]} hidden size.")
            training_scores, output_dict = train_and_test_model(**analysis_config, device=device)

            all_training_scores.append(training_scores)
            all_outputs.append(output_dict)

        # save results to csv
        df = pd.DataFrame(all_outputs)
        df['nfst_sizes'] = nfst_sizes[0]
        df.to_csv(os.path.join(model_folder, 'nfst_sizes.csv'), index=False)

        np.save(os.path.join(model_folder, 'training_scores.npy'), np.stack(all_training_scores))


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


