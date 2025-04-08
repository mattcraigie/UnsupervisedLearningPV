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

    assert analysis_type in ['sensitivity', 'data_scaling', 'nfst_sizes',
                              'noise', 'spiral', 'ratio', 'scale'], \
        f"Analysis type {analysis_type} not recognized."

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

    if analysis_type == 'data_scaling':
        key_1 = 'training_kwargs'
        key_2 = 'num_train_val_mocks'
    elif analysis_type == 'nfst_sizes':
        key_1 = 'model_kwargs'
        key_2 = 'subnet_hidden_sizes'
    elif analysis_type == 'sensitivity':
        key_1 = 'mock_kwargs'
        key_2 = 'ratio_left'
    elif analysis_type == 'noise':
        key_1 = 'mock_kwargs'
        key_2 = 'poisson_noise_level'
    elif analysis_type == 'spiral':
        key_1 = 'mock_kwargs'
        key_2 = 'total_num'
    elif analysis_type == 'ratio':
        key_1 = 'mock_kwargs'
        key_2 = 'ratio_left'
    elif analysis_type == 'scale':
        key_1 = 'mock_kwargs'
        key_2 = 'scales_min'
    else:
        raise ValueError(f"Analysis type {analysis_type} not recognized.")

    variable_features = analysis_config[key_1][key_2]

    outputs = []
    for i in range(len(variable_features)):

        analysis_folder = os.path.join(model_folder, f'{analysis_type}_{i}')
        os.makedirs(analysis_folder, exist_ok=True)
        analysis_config['output_root'] = analysis_folder

        analysis_config[key_1][key_2] = variable_features[i]

        logging.info(f"Running with {key_2}={variable_features[i]}")
        output_dict = train_and_test_model(**analysis_config, device=device)

        outputs.append(output_dict)

    # save results to csv
    all_val_scores = np.stack([output['val_scores'] for output in outputs])
    np.save(os.path.join(model_folder, 'val_scores.npy'), all_val_scores)

    # same for test scores
    all_test_scores = np.stack([output['test_scores'] for output in outputs])
    np.save(os.path.join(model_folder, 'test_scores.npy'), all_test_scores)

    # make test score df
    df_columns = [i[0] for i in variable_features] if analysis_type == 'nfst_sizes' else variable_features
    test_scores_df = pd.DataFrame(data=all_test_scores.T, columns=df_columns)

    # make summary df with mean, std, min, max, 1st, 2nd and 3rd quartiles of test scores
    summary_df = test_scores_df.describe().T
    summary_df.to_csv(os.path.join(model_folder, 'summary.csv'))

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


