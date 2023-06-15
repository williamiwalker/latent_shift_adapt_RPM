'''
create samples of discrete observations
'''
import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys, json, os, pickle

from latent_shift_adapt_RPM.utils.util_functions import save_checkpoint, small_discrete_example_alt, permute_latent, \
    posterior_counts, posterior_averages, load_model, sigmoid, observed_conditional, predict_q_y_x, RMSE


################################################################################################################
## Get Arguments
################################################################################################################

# just check if cuda is available
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = 'cpu'#torch.device("cuda" if cuda else "cpu")

# check if using CUDA
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# get sizes of datasets
ARG_FILE_NAME = 'arguments_cifar_partial_RPM_1.json'
parent_folder = '/your/directory' # directory that contains arg_files and to save outputs in
ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME
print(ARGUMENT_FILE)
with open(ARGUMENT_FILE) as json_file:
    ARGS = json.load(json_file)


created_sample_sizes = []
for run in ARGS.keys():

    paramDict = ARGS[run]

    OUTPUT_FOLDER = parent_folder + 'shared_data_4/'

    if os.path.exists(OUTPUT_FOLDER):
        print('overwrite')
    else:
        os.makedirs(OUTPUT_FOLDER)

    num_obs = paramDict['num_observations']
    if num_obs in created_sample_sizes:
        continue
    else:
        print('making dataset of size ',int(paramDict['num_observations']))
        source_obs, target_obs, true_full_joint, true_target_joint = small_discrete_example_alt(
            int(paramDict['num_observations']), device)

        samples = {'source_obs':source_obs, 'target_obs':target_obs, 'true_full_joint':true_full_joint,
                   'true_target_joint':true_target_joint}

        # write saved values to pickle file
        pickle.dump(samples, open(OUTPUT_FOLDER + str(int(num_obs))+'_samples.pkl', 'wb'))

        created_sample_sizes = created_sample_sizes + [num_obs]



