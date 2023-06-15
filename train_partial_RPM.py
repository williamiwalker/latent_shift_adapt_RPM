'''
test RPM used to learn q(Y|X) after latent shift
X and W observations are images
'''
import torch
import numpy as np

from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader

import sys, json, os, pickle, math
# sys.path.append('/nfs/gatsbystor/williamw/git/latent_shift_adapt_RPM/')

from models.partial_RPM import partial_RPM
from plotting.utils_plots import plot_loss, plot_digits, plot_var_probs, plot_factor_probs_together
from utils.util_functions import save_checkpoint
from utils.util_functions import save_checkpoint, permute_latent, posterior_counts, posterior_averages,\
    load_model, sigmoid, observed_conditional, predict_q_y_x, RMSE, small_discrete_example_alt,\
    MNIST_batches_with_replacement, cifar_batches_with_replacement, create_entropy_tmp_schd, create_prior_tmp_schd



################################################################################################################
## Get Arguments
################################################################################################################

SLURM_ARRAY_TASK_ID = sys.argv[1]
print('SLURM_ARRAY_TASK_ID ', SLURM_ARRAY_TASK_ID)

ARG_FILE_NAME = 'arguments_cifar_partial_RPM_test.json'
parent_folder = '/your/directory/' # directory that contains arg_files and to save outputs in
data_folder = 'shared_data_3/' # directory of saved generated data
ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME
print(ARGUMENT_FILE)
with open(ARGUMENT_FILE) as json_file:
    ARGS = json.load(json_file)
    print('PARAMETERS ', ARGS[SLURM_ARRAY_TASK_ID])
    paramDict = ARGS[SLURM_ARRAY_TASK_ID]

OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/' + paramDict['SUB_FOLDER']
saveFolder = parent_folder + OUTPUT_FOLDER + '/'




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

################################################################################################################
## create observations and train
################################################################################################################

# get saved discrete datapoints
num_obs = int(paramDict['num_observations'])
if paramDict['saved_samples']:
    print('using saved samples')
    samples = pickle.load(open(parent_folder + data_folder + str(num_obs)+'_samples.pkl', 'rb'))
    source_obs, target_obs, true_full_joint, true_target_joint = samples['source_obs'], samples['target_obs'], \
                                                                 samples['true_full_joint'], samples['true_target_joint']
else:
    source_obs, target_obs, true_full_joint, true_target_joint = small_discrete_example_alt(num_obs, device)

# draw random image samples for loaded discrete datapoints (for X and W)
# make data into batched, shuffled dataloader
if paramDict['observation_source']['X'] == 'cifar10':
    if ('MNIST_samples' in paramDict) and (paramDict['MNIST_samples'] == 'replacement'):
        source_loader_shuffle, _ = cifar_batches_with_replacement(source_obs, target_obs, paramDict['batch_size'], device, sample_size=paramDict['replacement_size'], shuffle=True)
        source_loader_non_shuffle, target_loader_non_shuffle = cifar_batches_with_replacement(source_obs, target_obs, paramDict['batch_size'], device, sample_size=paramDict['replacement_size'], shuffle=False)
elif paramDict['observation_source']['X'] == 'MNIST':
    if ('MNIST_samples' in paramDict) and (paramDict['MNIST_samples'] == 'replacement'):
        source_loader_shuffle, _ = MNIST_batches_with_replacement(source_obs, target_obs, paramDict['batch_size'], device, sample_size=paramDict['replacement_size'], shuffle=True)
        source_loader_non_shuffle, target_loader_non_shuffle = MNIST_batches_with_replacement(source_obs, target_obs, paramDict['batch_size'], device, sample_size=paramDict['replacement_size'], shuffle=False)


# # show examples of the digits
# for batch_idx, observations in enumerate(source_loader_non_shuffle):
#     print('obs',observations['X'].shape)
#     plot_digits(saveFolder, 'X_observations', observations['X'], source_obs['X'])
#     plot_digits(saveFolder, 'W_observations', observations['W'], source_obs['W'])
#     # plot_digits(saveFolder, 'Q_X_observations', target_obs['X'], target_obs['true_X'])
#     break



# Initialize Model
model = partial_RPM(paramDict, saveFolder, device)

# check if there is a temperature schedule for entropy or prior
# entropy tmp is a factor in front of the variational entropy term of the free energy (should decay to 1.0)
# prior tmp is a factor in front of the KL between variational entropy and prior (should decay to 1.0)
entropy_tmp_schedule = create_entropy_tmp_schd(paramDict, device)
prior_tmp_schedule = create_prior_tmp_schd(paramDict, device)
print('entropy_tmp_schedule', entropy_tmp_schedule)

# calculate sample_q_y_x so you can compute RMSE during training (just for seeing learning)
target_x_obs = torch.argmax(target_obs['X'], dim=1)
true_q_y_x, num_samples = observed_conditional(['Y'], ['X'], target_obs, paramDict['observation_dims'])
sample_q_y_x = torch.permute(true_q_y_x, (1, 0))[target_x_obs]

# Fit model to data
lossTrack, testRMSE, testEpochs, runtime = model.fit(source_loader_shuffle, source_loader_non_shuffle,
    entropy_tmp_schedule=entropy_tmp_schedule, prior_tmp_schedule=prior_tmp_schedule,
    target_loader_non_shuffle=target_loader_non_shuffle, target_obs=target_obs, sample_q_y_x=sample_q_y_x)

# save time data and loss
timeDict = {'free_energy': (-lossTrack/len(source_obs['X'])), 'times': runtime}
with open(saveFolder + 'time_data.pkl', 'wb') as handle:
    pickle.dump(timeDict, handle)

# save model one last time
save_checkpoint(model, paramDict['epochs'], model.loss_tot, saveFolder)




################################################################################################################
## Plots
################################################################################################################

####################################################
# Find true parameter values for comparison
full_joint = torch.tensor(true_full_joint, dtype=torch.float)
p_u = torch.einsum('uycxw->u',full_joint)
p_u_x = torch.einsum('uycxw->ux',full_joint)
p_u_x = p_u_x/torch.sum(p_u_x,dim=0).unsqueeze(0)
p_u_w = torch.einsum('uycxw->uw',full_joint)
p_u_w = p_u_w/torch.sum(p_u_w,dim=0).unsqueeze(0)
p_c_xu = torch.einsum('uycxw->cxu',full_joint)
p_c_xu = p_c_xu/torch.sum(p_c_xu,dim=0).unsqueeze(0)
p_y_cu = torch.einsum('uycxw->ycu',full_joint)
p_y_cu = p_y_cu/torch.sum(p_y_cu,dim=0).unsqueeze(0)
true_mat = {'true_P_prior':p_u, 'recog_mat':{'X':p_u_x, 'W':p_u_w}, 'gen_mat':{'C':p_c_xu, 'Y':p_y_cu}}


# save values for json file
saved_values = {}

# plot loss
plot_loss(saveFolder, 'free_energy', lossTrack)
plot_loss(saveFolder, 'test_RMSE', testRMSE)

# add true_X and true_W to observation dims
obs_dims = paramDict['observation_dims']
obs_dims['true_X'] = obs_dims['X']
obs_dims['true_W'] = obs_dims['W']

# get variational posteriors p(U|Y,X,W,C)
q_variational_probs = model.q_variational.probs.clone().detach().cpu()#.cpu().numpy()
q_variational_log_probs = torch.log(q_variational_probs)
print('q_variational log probs',q_variational_log_probs)

# permute latents so they are the same of true latents
# permutation = permute_latent(q_variational_log_probs, source_obs['U'], paramDict['latent_dim'])
# print('permutation',permutation)
permutation = np.arange(obs_dims['U']) # don't worry about permuting
saved_values['permutation'] = permutation


# get joint prob p(U,Y,X,W,C)
saved_values['true_full_joint'] = true_full_joint
# find joint according to samples
sample_joint, num_samples = observed_conditional(['U','Y','C','X','W'], [], source_obs, obs_dims)
# joint_prob_uycxw = model.get_joint_MNIST(source_obs)
# saved_values['learn_full_joint'] = joint_prob_uycxw.cpu().numpy()
saved_values['sample_full_joint'] = sample_joint.cpu().numpy()
saved_values['num_samples'] = num_samples.cpu().numpy()


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

print('sampled P(U|X)',true_posterior_probs_X)
print('learned P(U|X)',learn_posterior_probs_X)
print('number of samples',num_samples)

# check prior p(U)
p_u = sigmoid(np.array([1.0,0.1,0.1]))
print('true prior',p_u)
p_prior = model.latent_prior.probs[0].clone().detach().cpu().numpy()[permutation]
print('learned prior',p_prior)
saved_values['true P(U)'] = p_u
saved_values['learn P(U)'] = p_prior


# train new prior Q(U)
q_prior_free_energy, q_prior_runtime = model.learn_prior_only(target_loader_non_shuffle)
save_checkpoint(model, paramDict['epochs'], model.loss_tot, saveFolder)

# plot loss
plot_loss(saveFolder, 'Q_prior_free_energy', q_prior_free_energy)

# Q(U|X) according to true samples
true_Q_U_X_matrix, sample_counts = observed_conditional(['U'], ['X'], target_obs, obs_dims)

# learned Q(U|X)
learn_Q_U_X_matrix = posterior_averages(target_obs, model.Q_var_posteriors.probs, ['X'], obs_dims, permutation)

print('learn Q(U|X)',learn_Q_U_X_matrix)
print('true  Q(U|X)',true_Q_U_X_matrix.T)
saved_values['sample Q(U|X)'] = true_Q_U_X_matrix.cpu().numpy()
saved_values['learn Q(U|X)'] = learn_Q_U_X_matrix.T.cpu().numpy()

# Q(U) learned and true
learn_Q_prior = model.target_latent_prior.probs[0].clone().detach().cpu().numpy()[permutation]
true_Q_prior = sigmoid(np.array([0.1,0.1,1.0]))

print('learn Q prior',learn_Q_prior)
print('true Q prior ',true_Q_prior)
saved_values['true Q(U)'] = true_Q_prior
saved_values['learn Q(U)'] = learn_Q_prior



# get P(Y|X,U) and P(C|X,U)
generated_factor_dict = model.get_generated_probs(target_loader_non_shuffle)
p_y_cu = generated_factor_dict['Y'].detach().clone().cpu() # this is just a tensor
p_c_xu = generated_factor_dict['C'].detach().clone().cpu() # this is a matrix for every X

# plot examples of variational posterior on target distribution Q
Q_posteriors = model.Q_var_posteriors.probs.clone().detach().cpu().numpy()
plot_var_probs(saveFolder, 'Q_var_posterior_examples', Q_posteriors, torch.argmax(target_obs['X'], dim=1),
               paramDict['observation_dims']['X'])

# plot examples of variational posterior on source distribution P
q_variational_probs = model.q_variational.probs.clone().detach().cpu()
plot_var_probs(saveFolder, 'varitational_posterior_examples', q_variational_probs, torch.argmax(source_obs['X'], dim=1),
               paramDict['observation_dims']['X'])

# plot examples of factors on source distribution P
factor_u_x = model.factors[1].probs.detach().cpu().numpy()
plot_var_probs(saveFolder, 'factor_u_x_examples', factor_u_x, torch.argmax(source_obs['X'], dim=1),
               paramDict['observation_dims']['X'])
factor_u_w = model.factors[0].probs.detach().cpu().numpy()
plot_var_probs(saveFolder, 'factor_u_w_examples', factor_u_w, torch.argmax(source_obs['W'], dim=1),
               paramDict['observation_dims']['W'])
plot_factor_probs_together(saveFolder, 'factor_u_x_over_x_and_w_examples', factor_u_x,
                           torch.argmax(source_obs['X'], dim=1), paramDict['observation_dims']['X'],
                           torch.argmax(source_obs['W'], dim=1), paramDict['observation_dims']['W'])


# test learned P(Y|U,C) to samples
true_P_Y_UC, num_samples = observed_conditional(['Y'], ['U','C'], source_obs, obs_dims)
print('true  P(Y|U,C)',true_P_Y_UC)
print('learn P(Y|U,C)',torch.permute(torch.permute(p_y_cu,(2,0,1))[permutation],(1,0,2)))
saved_values['sample  P(Y|U,C)'] = true_P_Y_UC.cpu().numpy()
saved_values['learn P(Y|U,C)'] = torch.permute(torch.permute(p_y_cu,(2,0,1))[permutation],(1,0,2)).cpu().numpy()


# test learned P(C|U,X) to samples
true_P_C_UX, num_samples = observed_conditional(['C'], ['U','X'], source_obs, obs_dims)
learn_p_c_ux_matrix = torch.permute(posterior_averages(target_obs, p_c_xu, ['X'], obs_dims, permutation), (1,2,0))
print('true  P(C|U,X)',true_P_C_UX)
print('learn P(C|U,X)',torch.permute(torch.permute(learn_p_c_ux_matrix,(1,0,2))[permutation],(1,0,2)))
saved_values['sample  P(C|U,X)'] = true_P_C_UX.cpu().numpy()
saved_values['learn P(C|U,X)'] = torch.permute(torch.permute(learn_p_c_ux_matrix,(1,0,2))[permutation],(1,0,2)).cpu().numpy()

# test Q(Y|X) to true
target_y_obs = torch.argmax(target_obs['Y'], dim=1)
target_x_obs = torch.argmax(target_obs['X'], dim=1)
p_y_uc = torch.permute(p_y_cu,(0,2,1))
q_y_x = predict_q_y_x(target_obs, model.Q_var_posteriors.probs.cpu(), p_y_uc.cpu(), p_c_xu.cpu())
# print('learn Q(Y|X)',q_y_x)
true_q_y_x, num_samples = observed_conditional(['Y'], ['X'], target_obs, obs_dims)
print('Q(Y|X) true - learn',torch.permute(true_q_y_x,(1,0))[target_x_obs]-q_y_x)

print('learn Q(Y|X) with samples',torch.sum(q_y_x * target_obs['Y'],dim=1))
learn_q_y_x = posterior_averages(target_obs, q_y_x, ['X'], obs_dims, torch.arange(obs_dims['Y'])).T
saved_values['sample  Q(Y|X)'] = true_q_y_x.cpu().numpy()
saved_values['learn Q(Y|X)'] = learn_q_y_x.cpu().numpy()
saved_values['true target full joint'] = true_target_joint


sample_q_y_x = torch.permute(true_q_y_x,(1,0))[target_x_obs]

# compute RMSE of predicted Q(Y|X) to true
rmse_q_y_x = RMSE(sample_q_y_x[:,0], q_y_x[:,0])
saved_values['Q(Y|X) RMSE'] = rmse_q_y_x
print('RMSE: ',rmse_q_y_x)

# write saved values to pickle file
pickle.dump(saved_values, open(saveFolder + 'saved_values.pkl','wb'))
