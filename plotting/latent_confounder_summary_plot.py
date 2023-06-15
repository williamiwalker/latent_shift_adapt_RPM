'''
test RPM and previous models on different metrics for predicting y on target
'''
import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys, json, os, pickle, glob
sys.path.append('/nfs/ghome/live/williamw/git')
sys.path.append('/home/william/mnt/gatsbystor')
sys.path.append('/home/william/git')
from latent_shift_adapt_RPM.models.partial_RPM import partial_RPM
from latent_shift_adapt_RPM.utils.util_functions import KL_div, conditional_KL, small_discrete_example_alt, observed_conditional, RMSE, \
    MNIST_batches, load_model, MNIST_batches_with_replacement, posterior_counts, posterior_averages, predict_q_y_x, \
    sigmoid, cifar_batches_with_replacement, evaluate_clf, log_loss64, soft_accuracy, soft_balanced_accuracy
from utils_plots import plot_metric

from pathlib import Path

import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

################################################################################################################
## Get Arguments
################################################################################################################

ARG_FILE_NAME = 'arguments_cifar_partial_RPM_1.json'

parent_folder = '/nfs/gatsbystor/williamw/latent_confounder/'
# parent_folder = '/home/william/mnt/gatsbystor/latent_confounder/'
data_folder = parent_folder + 'shared_data_3/'
ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME
print(ARGUMENT_FILE)
with open(ARGUMENT_FILE) as json_file:
    ARGS = json.load(json_file)


evals_sklearn = {
    'cross-entropy': log_loss64,
    'accuracy': soft_accuracy,
    'balanced_accuracy': soft_balanced_accuracy,
    'auc': sklearn.metrics.roc_auc_score
}


# OUTPUT_FOLDER = 'cifar_partial_RPM_1'
NN_folder = 'shared_data_3'



# get all files in folder
full_LAT_df = pd.DataFrame({})
skipped_runs = []
prev_num_obs = -100
for run in ARGS.keys():

    paramDict = ARGS[run]
    OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/' + paramDict['SUB_FOLDER']+ '/'
    saveFolder = parent_folder + OUTPUT_FOLDER + '/'
    plot_folder = parent_folder + paramDict['MAIN_FOLDER'] + '/'

    if not(os.path.isfile(saveFolder + 'learned_model.pth')):
        continue

    print('run ',run)

    if (('saved_samples' in paramDict) and (paramDict['saved_samples']) and not(prev_num_obs == int(paramDict['num_observations']))):
        print('using saved samples')
        num_obs = int(paramDict['num_observations'])
        prev_num_obs = num_obs
        samples = pickle.load(
            open(data_folder + str(int(num_obs)) + '_samples.pkl', 'rb'))
        source_obs, target_obs, true_full_joint, true_target_joint = samples['source_obs'], samples['target_obs'], \
                                                                     samples['true_full_joint'], samples[
                                                                         'true_target_joint']


    # grab the saved metric values
    if os.path.isfile(saveFolder + 'saved_metrics.pkl'):
        saved_metrics = pickle.load(open(saveFolder + 'saved_metrics.pkl', 'rb'))
        full_LAT_df = full_LAT_df.append({'method':'RPM','eval_set':'eval_on_target',
                'cross-entropy':saved_metrics['cross-entropy']['eval_on_target'], 'accuracy':saved_metrics['accuracy']['eval_on_target'],
                'balanced_accuracy': saved_metrics['balanced_accuracy']['eval_on_target'], 'auc':saved_metrics['auc']['eval_on_target'],
                'iteration':run, 'x_dim':paramDict['num_observations']}, ignore_index=True)
        print('saved metrics RPM', saved_metrics)
    else:
        skipped_runs = skipped_runs + [saveFolder]


    if not os.path.isfile(data_folder + run + 'saved_metrics.pkl'):
        ################################################################################################################
        ## Evaluation metrics
        ################################################################################################################
        evals_sklearn = {
            'cross-entropy': log_loss64,
            'accuracy': soft_accuracy,
            'balanced_accuracy': soft_balanced_accuracy,
            'auc': sklearn.metrics.roc_auc_score
        }

        NN_saved_values = pickle.load(open(data_folder + run + '/saved_values.pkl', 'rb'))
        NN_q_y_x = torch.tensor(NN_saved_values['NN Q(Y|X)'])

        data_dict_all = {'target': {'test': {'y': torch.argmax(target_obs['Y'], axis=1).cpu().numpy()}}}
        result_NN = evaluate_clf(evals_sklearn, NN_q_y_x, data_dict_all)

        # write saved values to pickle file
        pickle.dump(result_NN, open(data_folder + run + '/saved_metrics.pkl', 'wb'))

        full_LAT_df = full_LAT_df.append({'method': 'ERM-SOURCE', 'eval_set': 'eval_on_target',
                                          'cross-entropy': result_NN['cross-entropy']['eval_on_target'],
                                          'accuracy': result_NN['accuracy']['eval_on_target'],
                                          'balanced_accuracy': result_NN['balanced_accuracy']['eval_on_target'],
                                          'auc': result_NN['auc']['eval_on_target'],
                                          'iteration': run, 'x_dim': paramDict['num_observations']},
                                         ignore_index=True)

    else:
        saved_metrics = pickle.load(open(data_folder + run + 'saved_metrics.pkl', 'rb'))

        full_LAT_df = full_LAT_df.append({'method': 'ERM-SOURCE', 'eval_set': 'eval_on_target',
                                          'cross-entropy': saved_metrics['cross-entropy']['eval_on_target'],
                                          'accuracy': saved_metrics['accuracy']['eval_on_target'],
                                          'balanced_accuracy': saved_metrics['balanced_accuracy']['eval_on_target'],
                                          'auc': saved_metrics['auc']['eval_on_target'],
                                          'iteration': run, 'x_dim': paramDict['num_observations']},
                                         ignore_index=True)




plot_metric(plot_folder, 'dataset_size', full_LAT_df, x_axis='dataset size')
print('skipped runs',skipped_runs)