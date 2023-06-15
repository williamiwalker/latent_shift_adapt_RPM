'''
test RPM used to learn q(Y|X) after latent shift
X observation is a vector
'''
import torch
import numpy as np

import sys, json, os, pickle, math

from models.partial_RPM import partial_RPM
from utils.util_functions import save_checkpoint,  permute_latent, posterior_counts, posterior_averages,\
    sigmoid, observed_conditional, predict_q_y_x, RMSE, extract_from_df_nested, vector_x_dataset, evaluate_clf,\
    log_loss64, soft_accuracy, soft_balanced_accuracy, read_dataframe_into_dict, create_entropy_tmp_schd, create_prior_tmp_schd
from plotting.utils_plots import plot_loss, plot_var_probs, plot_factor_probs_together

import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


################################################################################################################
## Get Arguments
################################################################################################################
num_runs = 10
arg = int(sys.argv[1])
x_dim = int((arg//num_runs)*2 + 2)
run = int(arg%num_runs)


folder_id = '/your/data/directory'
parent_folder = '/your/directory/to/save/outputs/'


OUTPUT_FOLDER = 'RPM_latent_adapt_test/x_dim_' + str(x_dim) + '_' + str(run) # name of the fdirectory to save your outputs
saveFolder = parent_folder + OUTPUT_FOLDER + '/'

# file names to get source and target data
filename_source = str(x_dim) + "_synthetic_multivariate_num_samples_10000_w_coeff_3_p_u_0_0.9.csv"
filename_target = str(x_dim) + "_synthetic_multivariate_num_samples_10000_w_coeff_3_p_u_0_0.1.csv"


# for MNIST partial RPM
paramDict = {
# 'MAIN_FOLDER': SUBFOLDER_NAME,
'observation_names': ['W','X'], # names of observations
'observation_dims': {'W':2, 'X':int(x_dim), 'Y':2, 'C': 2, 'U':2}, # dimensions of each observation
'observation_source': {'X':'vector', 'W':'discrete', 'Y':'discrete', 'C':'discrete'}, # which observations are images
'factor_conditioned':{'W':None, 'X':None, 'Y':'C', 'C':'X'}, # the conditioned variable for each observation
'observation_generated': ['C', 'Y'], # dimensions of the variables that condition the factors
'latent_dim': 2, # number of hidden discrete dim
'batch_epochs': 20, #2000, # number of epochs of batch training
'epochs': 20,#2000, # number of epochs to train for
'recog_learn_rate': 1e-4, # learning rate #was 1e-3
'gen_learn_rate': 1e-4, # learning rate for generated variables
'prior_learn_rate': 1e-3, # learning rate for prior
'target_prior_learn_rate': 1e-3, # learning rate for target prior
'saved_samples': True, # use the samples saved. same samples for each run with different initialization
'batch_size': 1000,
'prior_train_delay': 0, # number epochs to wait before training prior
'use_batching': True, # learn with batches
'exact_EM': False, # set to true if M step should also be exact
'dropout': False, # use dropout in factor neural network
'variational_learning_rate': 1e-2, # learning rate for sudo gradient in E step
'entropy_tmp_schedule': (0.2, 5.0), # temperature on entropy term, (percent to finish, start temperature)
# 'prior_tmp_schedule': (0.3, 5.0), # temperature on prior term, (percent to finish, start temperature)
'recog_net': 'MLP', # 'small_MNIST', #'small_MNIST' 'default'# specify which NN to use in recognition network
'plot_freq': 10 # how often (in epochs) to save model and make loss plot
}



# just check if cuda is available
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")

# check if using CUDA
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

if os.path.exists(saveFolder):
    print('overwrite')
else:
    os.makedirs(saveFolder)

# Data type: float64 / float32
data_type = torch.float32
torch.set_default_dtype(data_type)

################################################################################################################
## create observations and train
################################################################################################################


# read source and target data
source_obs, target_obs, num_obs, data_dict_all = read_dataframe_into_dict(folder_id, filename_source, filename_target,
                                                                          data_type, 'train')
paramDict['num_observations'] = num_obs

# put data into dataloader to pass to model
source_loader_shuffle, target_loader_non_shuffle = vector_x_dataset(source_obs, target_obs, batch_size=paramDict['batch_size'],shuffle=True)
source_loader_non_shuffle, _ = vector_x_dataset(source_obs, target_obs, batch_size=paramDict['batch_size'],shuffle=False)


# Init Model
model = partial_RPM(paramDict, saveFolder, device)


# check if there is a temperature schedule for entropy or prior
# entropy tmp is a factor in front of the variational entropy term of the free energy (should decay to 1.0)
# prior tmp is a factor in front of the KL between variational entropy and prior (should decay to 1.0)
entropy_tmp_schedule = create_entropy_tmp_schd(paramDict, device)
prior_tmp_schedule = create_prior_tmp_schd(paramDict, device)
print('entropy_tmp_schedule', entropy_tmp_schedule)

# calculate sample_q_y_x so you can compute RMSE during training
target_x_obs = torch.argmax(target_obs['X'], dim=1)

# Fit model
lossTrack, testRMSE, testEpochs, runtime = model.fit(source_loader_shuffle, source_loader_non_shuffle,
    entropy_tmp_schedule=entropy_tmp_schedule, prior_tmp_schedule=prior_tmp_schedule,
    target_loader_non_shuffle=target_loader_non_shuffle, target_obs=target_obs)#, sample_q_y_x=sample_q_y_x)

# save time data and loss
timeDict = {'free_energy': (-lossTrack/len(source_obs['X'])), 'times': runtime}
with open(saveFolder + 'time_data.pkl', 'wb') as handle:
    pickle.dump(timeDict, handle)

save_checkpoint(model, paramDict['epochs'], model.loss_tot, saveFolder)




################################################################################################################
## Plots
################################################################################################################

evals_sklearn = {
    'cross-entropy': log_loss64,
    'accuracy': soft_accuracy,
    'balanced_accuracy': soft_balanced_accuracy,
    'auc': sklearn.metrics.roc_auc_score
}

# save values for json file
saved_values = {}

# plot loss
plot_loss(saveFolder, 'free_energy', lossTrack)
plot_loss(saveFolder, 'test_RMSE', testRMSE)

# add true_X and true_W to observation dims
obs_dims = paramDict['observation_dims']
obs_dims['true_X'] = obs_dims['X']
obs_dims['true_W'] = obs_dims['W']

# # plot variational posteriors p(U|Y,X,W,C)
q_variational_probs = model.q_variational.probs.clone().detach().cpu()#.cpu().numpy()
q_variational_log_probs = torch.log(q_variational_probs)
print('q_variational log probs',q_variational_log_probs)

# permutation = permute_latent(q_variational_log_probs, source_obs['U'], paramDict['latent_dim'])
# print('permutation',permutation)
permutation = np.arange(obs_dims['U'])
saved_values['permutation'] = permutation

# find sampled posterior of p(U|X)
true_posterior_probs_X, num_samples = posterior_counts(source_obs, ['X'], obs_dims)
learn_posterior_probs_X = posterior_averages(source_obs, q_variational_probs, ['X'], obs_dims, permutation)
saved_values['sample P(U|X)'] = true_posterior_probs_X.T
saved_values['learn P(U|X)'] = learn_posterior_probs_X.T.cpu().numpy()

# find sampled posterior of p(U|W)
true_posterior_probs_W, num_samples = posterior_counts(source_obs, ['W'], obs_dims)
learn_posterior_probs_W = posterior_averages(source_obs, q_variational_probs, ['W'], obs_dims, permutation)
saved_values['sample P(U|W)'] = true_posterior_probs_W.T
saved_values['learn P(U|W)'] = learn_posterior_probs_W.T.cpu().numpy()
print('learn P(U|W)',learn_posterior_probs_W)
print('true P(U|W)',true_posterior_probs_W)

# print('sampled P(U|X)',true_posterior_probs_X)
# print('learned P(U|X)',learn_posterior_probs_X)
print('number of samples',num_samples)

# check prior p(U)
p_u = sigmoid(np.array([1.0,0.1,0.1]))
# print('true prior',p_u)
p_prior = model.latent_prior.probs[0].clone().detach().cpu().numpy()[permutation]
print('learned prior',p_prior)
saved_values['true P(U)'] = p_u
saved_values['learn P(U)'] = p_prior


# train new prior Q(U)
q_prior_free_energy, q_prior_runtime = model.learn_prior_only(target_loader_non_shuffle)
save_checkpoint(model, paramDict['epochs'], model.loss_tot, saveFolder)

# plot loss
plot_loss(saveFolder, 'Q_prior_free_energy', q_prior_free_energy)

# learned Q(U|X)
learn_Q_U_X_matrix = posterior_averages(target_obs, model.Q_var_posteriors.probs, ['X'], obs_dims, permutation)
saved_values['learn Q(U|X)'] = learn_Q_U_X_matrix.T.cpu().numpy()

# learned Q(U)
learn_Q_prior = model.target_latent_prior.probs[0].clone().detach().cpu().numpy()[permutation]
true_Q_prior = sigmoid(np.array([0.1,0.1,1.0]))
print('learn Q prior',learn_Q_prior)
saved_values['true Q(U)'] = true_Q_prior
saved_values['learn Q(U)'] = learn_Q_prior



# get P(Y|X,U) and P(C|X,U)
generated_factor_dict = model.get_generated_probs(target_loader_non_shuffle)
p_y_cu = generated_factor_dict['Y'].detach().clone().cpu() # this is just a tensor
p_c_xu = generated_factor_dict['C'].detach().clone().cpu() # this is a matrix for every X
print('p_c_xu', p_c_xu[:30])
print('p_y_cu', p_y_cu)

# plot examples of variational posterior on target distribution Q
Q_posteriors = model.Q_var_posteriors.probs.clone().detach().cpu().numpy()

# plot examples of variational posterior on source distribution P
q_variational_probs = model.q_variational.probs.clone().detach().cpu()

# plot examples of factors on source distribution P
factor_u_x = model.factors[1].probs.detach().cpu().numpy()
factor_u_w = model.factors[0].probs.detach().cpu().numpy()



# test learned P(Y|U,C) to samples
true_P_Y_UC, num_samples = observed_conditional(['Y'], ['U','C'], source_obs, obs_dims)
saved_values['sample  P(Y|U,C)'] = true_P_Y_UC.cpu().numpy()
saved_values['learn P(Y|U,C)'] = torch.permute(torch.permute(p_y_cu,(2,0,1))[permutation],(1,0,2)).cpu().numpy()


# test learned P(C|U,X) to samples
true_P_C_UX, num_samples = observed_conditional(['C'], ['U','X'], source_obs, obs_dims)
learn_p_c_ux_matrix = torch.permute(posterior_averages(target_obs, p_c_xu, ['X'], obs_dims, permutation), (1,2,0))
saved_values['sample  P(C|U,X)'] = true_P_C_UX.cpu().numpy()
saved_values['learn P(C|U,X)'] = torch.permute(torch.permute(learn_p_c_ux_matrix,(1,0,2))[permutation],(1,0,2)).cpu().numpy()

# test Q(Y|X) to true
target_y_obs = torch.argmax(target_obs['Y'], dim=1)
target_x_obs = torch.argmax(target_obs['X'], dim=1)
p_y_uc = torch.permute(p_y_cu,(0,2,1))
q_y_x = predict_q_y_x(target_obs, model.Q_var_posteriors.probs.cpu(), p_y_uc.cpu(), p_c_xu.cpu())
true_q_y_x, num_samples = observed_conditional(['Y'], ['X'], target_obs, obs_dims)

learn_q_y_x = posterior_averages(target_obs, q_y_x, ['X'], obs_dims, torch.arange(obs_dims['Y'])).T
saved_values['sample  Q(Y|X)'] = true_q_y_x.cpu().numpy()
saved_values['learn Q(Y|X)'] = learn_q_y_x.cpu().numpy()


sample_q_y_x = torch.permute(true_q_y_x,(1,0))[target_x_obs]

# compute RMSE of predicted Q(Y|X) to true
rmse_q_y_x = RMSE(sample_q_y_x[:,0], q_y_x[:,0])
saved_values['Q(Y|X) RMSE'] = rmse_q_y_x
print('RMSE: ',rmse_q_y_x)

# write saved values to pickle file
pickle.dump(saved_values, open(saveFolder + 'saved_values.pkl','wb'))


################################################################################################################
## Evaluation metrics
################################################################################################################

# read source and target data (for test set)
source_obs, target_obs, num_obs, data_dict_all = read_dataframe_into_dict(folder_id, filename_source, filename_target,
                                                                          data_type, 'test')
paramDict['num_observations'] = num_obs

source_loader_shuffle, target_loader_non_shuffle = vector_x_dataset(source_obs, target_obs, batch_size=paramDict['batch_size'],shuffle=True)
source_loader_non_shuffle, _ = vector_x_dataset(source_obs, target_obs, batch_size=paramDict['batch_size'],shuffle=False)

# Fit model
model.fit_params['epochs'] = 1
model.num_obs = source_obs['num_obs']
lossTrack, testRMSE, testEpochs, runtime = model.fit(source_loader_shuffle, source_loader_non_shuffle,
    target_loader_non_shuffle=target_loader_non_shuffle, target_obs=target_obs, sample_q_y_x=sample_q_y_x, save=False)

# get test dataset generative predictions
generated_factor_dict = model.get_generated_probs(target_loader_non_shuffle)
p_y_cu = generated_factor_dict['Y'].detach().clone().cpu() # this is just a tensor
p_c_xu = generated_factor_dict['C'].detach().clone().cpu() # this is a matrix for every X

print('q var shape',model.q_variational.probs.clone().detach().cpu().shape)
source_p_y_x = predict_q_y_x(source_obs, model.q_variational.probs.clone().detach().cpu(), p_y_uc.cpu(), p_c_xu.cpu())

q_prior_free_energy, q_prior_runtime = model.learn_prior_only(target_loader_non_shuffle)
print('Q var posteriors shape',model.Q_var_posteriors.probs.cpu().shape)
p_y_uc = torch.permute(p_y_cu,(0,2,1))
q_y_x = predict_q_y_x(target_obs, model.Q_var_posteriors.probs.cpu(), p_y_uc.cpu(), p_c_xu.cpu())



print('prediction q_y_x',q_y_x[:30])
result = evaluate_clf(evals_sklearn, q_y_x, data_dict_all, y_pred_source=source_p_y_x)
print('result',result)

# write saved values to pickle file
pickle.dump(result, open(saveFolder + 'saved_metrics.pkl','wb'))