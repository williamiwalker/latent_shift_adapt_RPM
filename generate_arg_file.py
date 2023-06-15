# This creates a json file that has a list of arguments that each job in a slurm job array will read
import json
import os



################################
# ARGUMENTS
################################
SUBFOLDER_NAME = 'cifar_partial_RPM_test'




ARG_FILE_NAME = 'arguments_' + SUBFOLDER_NAME +'.json'
ARGUMENT_FILE = '/your/directory/to/arg_files/'+ARG_FILE_NAME # where your arg_file is saved

COMMENTS = 'using deterministic u->w and u->x mapping. making dim of x=2. USING ENTORPY TEMPERATURE PARAMETER! train partial RPM. USING ADAM'







# # for partial RPM
# mainArgs = {
# 'MAIN_FOLDER': SUBFOLDER_NAME,
# 'observation_names': ['W','X'], # names of observations
# 'observation_dims': {'W':3, 'X':2, 'Y':2, 'C': 3, 'U':3}, # dimensions of each observation
# 'observation_source': {'X':'discrete', 'W':'discrete', 'Y':'discrete', 'C':'discrete'}, # which observations are images
# 'factor_conditioned':{'W':None, 'X':None, 'Y':'C', 'C':'X'}, # the conditioned variable for each observation
# 'observation_generated': ['C', 'Y'], # dimensions of the variables that condition the factors
# 'latent_dim': 3, # number of hidden discrete dim
# 'batch_epochs': 0, # number of epochs of batch training
# 'epochs': 10000, # number of epochs to train for
# 'recog_learn_rate': 1e-2, # learning rate
# 'gen_learn_rate': 1e-2, # learning rate for generated variables
# 'prior_learn_rate': 1e-2, # learning rate for prior
# 'saved_samples': True, # use the samples saved. same samples for each run with different initialization
# 'batch_size': 100,
# 'prior_train_delay': 0, # number epochs to wait before training prior
# 'use_batching': False, # learn with batches
# 'exact_EM': False, # set to true if M step should also be exact
# 'variational_learning_rate': 1e-1, # learning rate for sudo gradient in E step
# 'entropy_tmp_schedule': (0.3, 5.0), # temperature on entropy term, (percent to finish, start temperature)
# # 'prior_tmp_schedule': (0.3, 5.0), # temperature on prior term, (percent to finish, start temperature)
# }

# for MNIST partial RPM
mainArgs = {
'MAIN_FOLDER': SUBFOLDER_NAME,
'observation_names': ['W','X'], # names of observations
'observation_dims': {'W':3, 'X':2, 'Y':2, 'C': 3, 'U':3}, # dimensions of each observation
# 'observation_source': {'X':'MNIST', 'W':'MNIST', 'Y':'discrete', 'C':'discrete'}, # which observations are images
'observation_source': {'X':'cifar10', 'W':'cifar10', 'Y':'discrete', 'C':'discrete'}, # which observations are images
'factor_conditioned':{'W':None, 'X':None, 'Y':'C', 'C':'X'}, # the conditioned variable for each observation
'observation_generated': ['C', 'Y'], # dimensions of the variables that condition the factors
'latent_dim': 3, # number of hidden discrete dim
'batch_epochs': 20000, # number of epochs of batch training
'epochs': 20000, # number of epochs to train for
'recog_learn_rate': 1e-3, # learning rate
'gen_learn_rate': 1e-4, # learning rate for generated variables
'prior_learn_rate': 1e-3, # learning rate for prior
'target_prior_learn_rate': 1e-6, # learning rate for target prior
'saved_samples': True, # use the samples saved. same samples for each run with different initialization
'batch_size': 2000,
'prior_train_delay': 0, # number epochs to wait before training prior
'use_batching': True, # learn with batches
'exact_EM': False, # set to true if M step should also be exact
'dropout': False, # use dropout in factor neural network
'variational_learning_rate': 1e-2, # learning rate for sudo gradient in E step
'entropy_tmp_schedule': (0.15, 5.0), # temperature on entropy term, (percent to finish, start temperature)
# 'prior_tmp_schedule': (0.3, 5.0), # temperature on prior term, (percent to finish, start temperature)
'recog_net': 'cifarNet', # 'small_MNIST', #'small_MNIST' 'default'# specify which NN to use in recognition network
'MNIST_samples': 'replacement',  # if set to replacement, sample MNIST digits with replacement
'replacement_size': 5000, # number of samples to choose from in MNIST digit training set
'plot_freq': 10 # how often (in epochs) to save model and make loss plot
}





# NUM_RUNS = 20
num_observations = [5e2,1e3,2e3,5e3,1e4,2e4,5e4,1e5,2e5,5e5,1e6]
num_runs = 5
arguments = {}

job_index = 0
for indm in range(len(num_observations)):
    for indr in range(num_runs):
        currDict = mainArgs.copy()
        currDict['SUB_FOLDER'] = 'run_'+str(job_index)
        currDict['COMMENTS'] = COMMENTS
        currDict['num_observations'] = num_observations[indm]
        arguments[job_index] = currDict
        job_index += 1

print('sbatch --array=0-'+ str(job_index-1) + ' cpu_.sbatch')





if os.path.exists(ARGUMENT_FILE):
    print('overwrite')
    # raise Exception('You tryina overwrite a folder that already exists. Dont waste my time, sucka')

with open(ARGUMENT_FILE, 'w') as f:
    json.dump(arguments, f, indent=4)


