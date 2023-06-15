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
from latent_shift_adapt_RPM.plotting.utils_plots import plot_metric

from pathlib import Path

import pandas as pd

################################################################################################################
## Get Arguments
################################################################################################################

num_runs = 10

# folder_id = '/nfs/gatsbystor/williamw/latent_confounder/HMM_shared_data_linear_0'
folder_id = '/nfs/gatsbystor/williamw/latent_confounder/shared_data_linear_3'
parent_folder = '/nfs/gatsbystor/williamw/latent_confounder/'
# # parent_folder = '/home/william/mnt/gatsbystor/latent_confounder/'

OUTPUT_FOLDER = 'latent_adapt_3'
saveFolder = parent_folder + OUTPUT_FOLDER + '/'
RPM_folder = 'RPM_latent_adapt_3'

# put all results of different metrics into one dataframe
all_x_dims = np.arange(2,22,2,dtype=int)
full_LAT_df = pd.DataFrame({})
skipped_runs = []
for x_dim in all_x_dims:
    for run in range(num_runs):
        RPM_OUTPUT_FOLDER = RPM_folder + '/x_dim_' + str(x_dim) + '_' + str(run)
        RPM_saveFolder = parent_folder + RPM_OUTPUT_FOLDER + '/'
        if os.path.isfile(RPM_saveFolder + 'saved_metrics.pkl'):
            saved_metrics = pickle.load(open(RPM_saveFolder + 'saved_metrics.pkl', 'rb'))
            full_LAT_df = full_LAT_df.append({'method':'RPM','eval_set':'eval_on_target',
                    'cross-entropy':saved_metrics['cross-entropy']['eval_on_target'], 'accuracy':saved_metrics['accuracy']['eval_on_target'],
                    'balanced_accuracy': saved_metrics['balanced_accuracy']['eval_on_target'], 'auc':saved_metrics['auc']['eval_on_target'],
                    'iteration':run, 'x_dim':x_dim}, ignore_index=True)
            full_LAT_df = full_LAT_df.append({'method':'RPM','eval_set':'eval_on_source',
                    'cross-entropy':saved_metrics['cross-entropy']['eval_on_source'], 'accuracy':saved_metrics['accuracy']['eval_on_source'],
                    'balanced_accuracy': saved_metrics['balanced_accuracy']['eval_on_source'], 'auc':saved_metrics['auc']['eval_on_source'],
                    'iteration':run, 'x_dim':x_dim}, ignore_index=True)
        else:
            skipped_runs = skipped_runs + [RPM_OUTPUT_FOLDER]

    LAT_OUTPUT_FOLDER = OUTPUT_FOLDER + '/x_dim_' + str(x_dim)
    LAT_saveFolder = parent_folder + LAT_OUTPUT_FOLDER + '/'

    LAT_df = pd.read_csv(LAT_saveFolder + 'method_comparison_table.csv')
    LAT_df['x_dim'] = np.ones(len(LAT_df),dtype=int) * x_dim
    full_LAT_df = pd.concat([full_LAT_df, LAT_df])

print('SKIPPED RUNS: ',skipped_runs)



plot_metric(saveFolder, 'x_dim_metric', full_LAT_df,x_axis='x observation size')