import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import sys

sys.path.append("./../")
from latent_shift_adapt_RPM.utils.util_functions import categorical_cross_entropy, save_checkpoint, \
    unnormalized_categorical_cross_entropy, observed_conditional, posterior_averages, predict_q_y_x, RMSE
from latent_shift_adapt_RPM.plotting.utils_plots import plot_loss
import time
from scipy.optimize import fsolve



# def give_index_obs(obs):
#     '''
#     give observations as indices if they are one-hot
#     '''
#     if len(obs[0].shape) == 0:
#         # only use indices
#         return obs
#     else:
#         # only use indices
#         return torch.argmax(obs, dim=1)


class Net(nn.Module):
    # Convolutional Neural Network shared across independent factors
    def __init__(self, latent_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4 * 4 * 20, 50)
        self.fc2 = nn.Linear(50, latent_dim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4 * 4 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)



def batch_net(net, observations, obs_name, cond_obs_name, obs_dim, latent_dim, batch_size, num_obs, device):
    '''
    pass observations to network net in batches
    '''
    # print('observation types', type(obs), type(cond_obs))
    if obs_name is None:
        outputs = torch.zeros(num_obs, obs_dim, latent_dim).to(device)
        batch_ind = 0
        for inputs in observations:
            cond_obs_batch = inputs[cond_obs_name].to(device)
            bsize = cond_obs_batch.shape[0]
            outputs[int(batch_ind * batch_size): int((batch_ind * batch_size) + bsize)] = net(None, cond_obs_batch)
            batch_ind += 1
        return outputs
    elif cond_obs_name is None:
        outputs = torch.zeros(num_obs, latent_dim).to(device)
        batch_ind = 0
        for b_num, inputs in enumerate(observations):
            obs_batch = inputs[obs_name].to(device)
            bsize = obs_batch.shape[0]
            outputs[int(batch_ind * batch_size): int((batch_ind * batch_size) + bsize)] = net(obs_batch)
            batch_ind += 1
        return outputs
    else:
        outputs = torch.zeros(num_obs, latent_dim).to(device)
        batch_ind = 0
        for inputs in observations:
            cond_obs_batch = inputs[cond_obs_name].to(device)
            obs_batch = inputs[obs_name].to(device)
            bsize = cond_obs_batch.shape[0]
            outputs[int(batch_ind * batch_size): int((batch_ind * batch_size) + bsize)] = net(obs_batch, cond_obs_batch)
            batch_ind += 1
        return outputs


class recog_matrix(nn.Module):
    def __init__(self, latent_dim, obs_dim, true_mat=None):
        super(recog_matrix, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.true_mat = true_mat
        if true_mat is None:
            rand_mat = (torch.rand(latent_dim, obs_dim) - 0.5) * 0.1 + 0.5
            rand_mat = rand_mat / torch.sum(rand_mat, dim=0).unsqueeze(0)
            self.mat = nn.Parameter(torch.log(rand_mat))
            # self.mat = nn.Parameter(torch.log(torch.rand(latent_dim, obs_dim)))
        else:
            self.mat = nn.Parameter(torch.log(true_mat))
        # self.mat = nn.Parameter(torch.log(torch.ones(latent_dim, obs_dim)))

    def forward(self, x):
        prob_mat = torch.exp(self.mat)
        prob_mat = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
        return torch.log(torch.einsum('ux,dx->du', prob_mat, x))
        # if self.true_mat is None:
        #     return F.log_softmax(self.fc1(x), dim=-1)
        # else:
        #     prob_mat = torch.exp(self.fc1)
        #     prob_mat = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
        #     return torch.log(torch.einsum('ux,dx->du',prob_mat,x))

    def get_prob_matrix(self):
        prob_mat = torch.exp(self.mat)
        prob_mat = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
        return prob_mat


class generated_matrix(nn.Module):
    def __init__(self, obs_dim, cond_dim, latent_dim, true_mat=None):
        super(generated_matrix, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.cond_dim = cond_dim
        self.true_mat = true_mat
        if self.true_mat is None:
            if self.cond_dim is None:
                rand_mat = (torch.rand(obs_dim, latent_dim) - 0.5) * 0.1 + 0.5
                rand_mat = rand_mat / torch.sum(rand_mat, dim=0).unsqueeze(0)
                self.mat = nn.Parameter(torch.log(rand_mat))
                # self.mat = nn.Parameter(torch.log(torch.rand(obs_dim, latent_dim)))
            else:
                rand_mat = (torch.rand(obs_dim, cond_dim, latent_dim) - 0.5) * 0.1 + 0.5
                rand_mat = rand_mat / torch.sum(rand_mat, dim=0).unsqueeze(0)
                self.mat = nn.Parameter(torch.log(rand_mat))
                # self.mat = nn.Parameter(torch.log(torch.rand(obs_dim, cond_dim, latent_dim)))
        else:
            self.mat = nn.Parameter(torch.log(true_mat))

    def forward(self, obs, cond_obs):
        prob_mat = torch.exp(self.mat)
        prob_mat = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
        if len(obs[0].shape) > 0:
            obs = torch.argmax(obs, dim=1)
        if self.cond_dim is None:
            return prob_mat[obs]
        else:
            if len(cond_obs[0].shape) > 0:
                cond_obs = torch.argmax(cond_obs, dim=1)
            return prob_mat[obs, cond_obs]

    def get_prob_matrix(self):
        prob_mat = torch.exp(self.mat)
        prob_mat = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
        return prob_mat


class generated_net(nn.Module):
    def __init__(self, obs_dim, latent_dim, data_source='MNIST', dropout=False, input_dim=None):
        super(generated_net, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.output_dim = int(obs_dim * latent_dim)
        if data_source == 'cifar10':
            self.net = cifarNet(self.output_dim, dropout=dropout)
        elif data_source == 'MNIST':
            self.net = MNIST_Net(self.output_dim, dropout=dropout)
        elif data_source == 'vector':
            self.net = MLP(num_inputs=input_dim, num_digits=self.output_dim)

    def forward(self, obs, cond_obs):
        output = self.net(cond_obs)
        if torch.sum(torch.isnan(output)) > 0:
            for name, param in self.net.named_parameters():
                print(name, param)
            raise Exception('got nan in generated net ', cond_obs.shape)
        if torch.sum(torch.isinf(output)) > 0:
            for name, param in self.net.named_parameters():
                print(name, param)
            raise Exception('got inf in generated net ', cond_obs.shape)

        if obs is None:
            return torch.permute(F.softmax(output.reshape(output.shape[0], self.latent_dim, self.obs_dim), dim=-1),
                                 (0, 2, 1))
        else:
            if len(obs[0].shape) > 0:
                obs = torch.argmax(obs, dim=1)
            return F.softmax(output.reshape(output.shape[0], self.latent_dim, self.obs_dim), dim=-1)[
                   torch.arange(len(obs)), :, obs]


class recognition_net(nn.Module):
    def __init__(self, latent_dim, input_dim=None, dropout=False, data_source='MNIST', recog_net='default'):
        super(recognition_net, self).__init__()
        self.latent_dim = latent_dim
        if data_source == 'cifar10':
            if recog_net == 'cifarNet':
                self.net = cifarNet(latent_dim, dropout=dropout)
            else:
                self.net = ResNet(latent_dim)
        elif data_source == 'MNIST':
            if recog_net == 'small_MNIST':
                print('using small MNIST recognition network')
                self.net = MNIST_small_net(latent_dim, dropout=dropout)
            else:
                self.net = MNIST_Net(latent_dim, dropout=dropout)
        elif data_source == 'vector':
            self.net = MLP(num_inputs=input_dim, num_digits=latent_dim)

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)



class MNIST_Net(nn.Module):
    def __init__(self, latent_dim, dropout=False):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, latent_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.dropout:
            print('using dropout 1')
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if self.dropout:
            print('using dropout 2')
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=-1)


class MNIST_small_net(nn.Module):
    def __init__(self, latent_dim, dropout=False):
        super(MNIST_small_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 8, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128, 20)
        self.fc2 = nn.Linear(20, latent_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.dropout:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class cifarNet(nn.Module):
    # Convolutional Neural Network shared across independent factors
    def __init__(self, num_digits, dropout=False):
        super(cifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5 * 5 * 20, 50)
        self.fc2 = nn.Linear(50, num_digits)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.dropout:
            # print('using dropout 1')
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 5 * 5 * 20)
        x = F.relu(self.fc1(x))
        if self.dropout:
            # print('using dropout 2')
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    # Convolutional Neural Network shared across independent factors
    def __init__(self, num_inputs, num_digits):
        super(MLP, self).__init__()
        self.num_inputs = num_inputs
        self.num_digits = num_digits
        self.fc1 = nn.Linear(num_inputs, 100)
        self.fc2 = nn.Linear(100, num_digits)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class partial_RPM(nn.Module):
    """
    Recognition-Parametrized Model (RPM) with discrete latents
    """

    def __init__(self, fit_params, save_folder, device,
                 recognition_network=None, logits_prior=None, q_variational=None, true_init=None):

        super(partial_RPM, self).__init__()

        self.latent_dim = int(fit_params['latent_dim'])  # dimension of U
        self.fit_params = fit_params  # dictionary of parameters
        self.obs_names = fit_params['observation_names']  #
        self.obs_dimensions = fit_params['observation_dims']
        self.obs_generated = fit_params['observation_generated']
        self.obs_source = fit_params['observation_source']
        self.obs_conditioned = fit_params['factor_conditioned']
        self.num_obs = int(fit_params['num_observations'])
        self.batch_size = fit_params['batch_size']
        self.loss_tot = None
        self.device = device
        self.save_folder = save_folder
        self.true_init = true_init
        self.target_logits_prior = None

        # Use dropout in recognition networks
        if 'dropout' in self.fit_params:
            self.dropout = self.fit_params['dropout']
        else:
            self.dropout = False

        # Initialize variational distribution q(U)
        if q_variational is None:
            self.init_variational()
        else:
            self.q_variational = q_variational

        # Initialize Factor Prior P(U)
        if logits_prior is None:
            self.init_prior()
            self.update_prior()
            self.init_target_prior()
        else:
            self.logits_prior = logits_prior
            self.update_prior()

        # Initialize Recognition Network f(U|X) and f(U|W)
        if recognition_network is None:
            self.init_recognition_network(self.dropout)
        else:
            self.recognition_network = recognition_network

        #
        self.init_generated_probs()

        # Init / Update factors
        self.factors = [None] * len(self.obs_names)

    def get_loss(self, entropy_tmp=1.0, prior_tmp=1.0):
        '''
        calculate free energy
        '''

        # Model
        prior = self.latent_prior
        factors = self.factors
        generated_factors = self.generated_factors
        variational = self.q_variational

        # Dimensions of the problem
        num_obs = self.num_obs
        num_factors = len(factors)

        # <log q>_q
        variational_entropy = - categorical_cross_entropy(variational, variational, device=self.device)

        # <log p>_q
        prior_xentropy = categorical_cross_entropy(variational, prior, device=self.device)

        # <log f(.|x)>_q
        factors_xentropy = \
            sum([categorical_cross_entropy(variational, factor_i, device=self.device) for factor_i in factors])

        # log p0
        loss_offset = len(prior_xentropy) * num_factors * torch.log(1 / torch.tensor([num_obs])).to(self.device)

        # < log \sum f(.|x)  >_q
        denominator_xentropy = sum(
            [categorical_cross_entropy(variational, f_i, device=self.device) for f_i in self.denominators])

        # < log p(c|u,x)> + < log p(y|c,u)>
        generated_xentropy = sum(
            [unnormalized_categorical_cross_entropy(variational, f_i_probs, device=self.device) for f_i_probs in
             generated_factors])

        # Free Energy / ELBO
        free_energy = \
            torch.sum(
                prior_tmp * variational_entropy + prior_tmp * prior_xentropy + generated_xentropy + factors_xentropy - denominator_xentropy) + loss_offset  ############## NO ENTROPY_TMP TERM!!!!!!!!!!!!

        return - free_energy / float(num_obs)


    def fit(self, obs_loader_shuffle, obs_loader_non_shuffle, entropy_tmp_schedule=None, prior_tmp_schedule=None,
            target_loader_non_shuffle=None, target_obs=None, sample_q_y_x=None, save=True):
        '''
        learn parameters of recognition networks from observations
        '''

        # Fit parameters
        learn_epochs = self.fit_params['epochs']
        batch_epochs = self.fit_params['batch_epochs']
        num_batches = len(obs_loader_shuffle)

        # if there is no entropy or prior temperature schedule, make all ones
        if entropy_tmp_schedule is None:
            entropy_tmp_schedule = torch.ones(learn_epochs).to(self.device)
        if prior_tmp_schedule is None:
            prior_tmp_schedule = torch.ones(learn_epochs).to(self.device)

        # Network Optimizers
        optimizer_prior = torch.optim.Adam([self.logits_prior], lr=(self.fit_params['prior_learn_rate']))
        optimizer_recog = torch.optim.Adam(self.recognition_network.parameters(),
                                           lr=(self.fit_params['recog_learn_rate']))
        optimizer_gen = torch.optim.Adam(self.generation_networks.parameters(), lr=(self.fit_params['gen_learn_rate']))

        # Init Loss
        loss_tot = np.zeros(learn_epochs)
        test_rate = self.fit_params['plot_freq']
        test_RMSE_tot = np.zeros(learn_epochs // test_rate)
        test_epochs = np.zeros(len(test_RMSE_tot))

        # for timing
        runtime = np.zeros(learn_epochs)
        startTime = time.time()

        for epoch in range(learn_epochs):

            entropy_tmp = entropy_tmp_schedule[epoch]
            prior_tmp = prior_tmp_schedule[epoch]

            if (epoch < batch_epochs) and (epoch < (learn_epochs - 1)):
                # use batches for learning
                batch_loss = 0.0

                for batch_idx, observations in enumerate(obs_loader_shuffle):
                    num_obs = len(observations['X'])

                    # Clear gradients for this training step
                    optimizer_prior.zero_grad()
                    optimizer_recog.zero_grad()
                    optimizer_gen.zero_grad()

                    start_t = time.time()

                    # Update prior
                    self.update_prior(num_obs=num_obs)

                    # Factors distributions are the output of neural network
                    self.update_factors(observations)

                    # Update unnormalized probabilities self.generated_factors according to observations (p(c|u,x) and p(y|c,u))
                    self.update_generated_factors(observations)

                    time_1 = time.time()

                    # E-Step
                    with torch.no_grad():
                        self.update_variational(entropy_tmp=entropy_tmp, prior_tmp=prior_tmp)

                    time_2 = time.time()

                    # Loss and backprop
                    loss = self.get_loss(entropy_tmp=entropy_tmp, prior_tmp=prior_tmp)
                    loss.backward(retain_graph=True)

                    # M-Step apply gradients
                    if epoch >= int(self.fit_params['prior_train_delay']):
                        optimizer_prior.step()
                    optimizer_recog.step()
                    optimizer_gen.step()

                    time_3 = time.time()

                    batch_loss += -loss.detach()


                loss_tot[epoch] = batch_loss
                print('Iteration :' + str(epoch + 1) + '/' + str(learn_epochs) + ' Loss ' + str(
                    -loss.detach().cpu().numpy()) + ' times: ' + str(round(time_1 - start_t, 3)) + ' ' + str(
                    round(time_2 - time_1, 3)) + ' ' + str(round(time_3 - time_2, 3)))

                # keep track of run time
                curTime = time.time()
                runtime[epoch] = curTime - startTime

            elif epoch == (learn_epochs - 1):
                # stop using batches and just do gradient decent
                print('last epoch')
                with torch.no_grad():

                    num_obs = self.num_obs

                    start_t = time.time()

                    # Update prior
                    self.update_prior(num_obs=num_obs)
                    print('updated prior')

                    # Factors distributions are the output of neural network
                    self.update_factors(obs_loader_non_shuffle)
                    print('updated factors')

                    # Update unnormalized probabilities self.generated_factors according to observations (p(c|u,x) and p(y|c,u))
                    self.update_generated_factors(obs_loader_non_shuffle)
                    print('updated gen factors')

                    time_1 = time.time()

                    # E-Step
                    self.update_variational()
                    print('updated variational')

                    time_2 = time.time()

                    # Loss
                    loss = self.get_loss()

                    time_3 = time.time()

                    loss_tot[epoch] = -loss.detach()
                    print('Iteration :' + str(epoch + 1) + '/' + str(learn_epochs) + ' Loss ' + str(
                        -loss.detach().cpu().numpy()) + ' times: ' + str(round(time_1 - start_t, 3)) + ' ' + str(
                        round(time_2 - time_1, 3)) + ' ' + str(round(time_3 - time_2, 3)))

                    # keep track of run time
                    curTime = time.time()
                    runtime[epoch] = curTime - startTime

            else:
                # stop using batches and just do gradient decent

                num_obs = self.num_obs

                # Clear gradients for this training step
                optimizer_prior.zero_grad()
                optimizer_recog.zero_grad()
                optimizer_gen.zero_grad()

                start_t = time.time()

                # Update prior
                self.update_prior(num_obs=num_obs)
                # print('cur prior', torch.exp(self.logits_prior) / torch.sum(torch.exp(self.logits_prior)))

                # Factors distributions are the output of neural network
                self.update_factors(obs_loader_non_shuffle)

                # Update unnormalized probabilities self.generated_factors according to observations (p(c|u,x) and p(y|c,u))
                self.update_generated_factors(obs_loader_non_shuffle)

                time_1 = time.time()

                # E-Step
                with torch.no_grad():
                    self.update_variational(entropy_tmp=entropy_tmp, prior_tmp=prior_tmp)
                    # self.update_variational_sudo_gradient(entropy_tmp=entropy_tmp,
                    #                                       prior_tmp=prior_tmp)

                time_2 = time.time()

                # Loss and Backprop
                loss = self.get_loss(entropy_tmp=entropy_tmp, prior_tmp=prior_tmp)
                loss.backward(retain_graph=True)

                # M-Step apply gradients
                if (epoch < (learn_epochs - 1)):
                    if epoch >= int(self.fit_params['prior_train_delay']):
                        optimizer_prior.step()
                    optimizer_recog.step()
                    optimizer_gen.step()

                time_3 = time.time()

                loss_tot[epoch] = -loss.detach()
                print('Iteration :' + str(epoch + 1) + '/' + str(learn_epochs) + ' Loss ' + str(
                    -loss.detach().cpu().numpy()) + ' times: ' + str(round(time_1 - start_t, 3)) + ' ' + str(
                    round(time_2 - time_1, 3)) + ' ' + str(round(time_3 - time_2, 3)))

                # scheduler.step(loss)

                # keep track of run time
                curTime = time.time()
                runtime[epoch] = curTime - startTime

            # check performance on target data
            if save and (target_loader_non_shuffle is not None) and (epoch % test_rate == 0) and (
                    sample_q_y_x is not None):
                print('checking RMSE')
                optimizer_prior.zero_grad()
                optimizer_recog.zero_grad()
                optimizer_gen.zero_grad()
                test_RMSE_tot[int(epoch // test_rate)] = self.learn_and_test_on_target(target_loader_non_shuffle,
                                                                                       target_obs, sample_q_y_x)
                # self.target_logits_prior.requires_grad = False

            # save model at checkpoint
            if save and (epoch % int(self.fit_params['plot_freq']) == 0):
                save_checkpoint(self, epoch, loss_tot, self.save_folder)
                # plot free energy
                plot_loss(self.save_folder, 'free_energy', loss_tot)

        optimizer_prior.zero_grad()
        optimizer_recog.zero_grad()
        optimizer_gen.zero_grad()

        self.loss_tot = loss_tot
        if save:
            save_checkpoint(self, epoch, loss_tot, self.save_folder)

        return self.loss_tot, test_RMSE_tot, test_epochs, runtime

    def learn_and_test_on_target(self, target_loader_non_shuffle, target_obs, sample_q_y_x):
        # train new prior Q(U)
        q_prior_free_energy, q_prior_runtime = self.learn_prior_only(target_loader_non_shuffle)
        with torch.no_grad():
            # get P(Y|X,U) and P(C|X,U)
            generated_factor_dict = self.get_generated_probs(target_loader_non_shuffle)
            p_y_cu = generated_factor_dict['Y'].detach().clone()  # this is just a tensor
            p_c_xu = generated_factor_dict['C'].detach().clone()  # this is a matrix for every X
            p_y_uc = torch.permute(p_y_cu, (0, 2, 1))
            q_y_x = predict_q_y_x(target_obs, self.Q_var_posteriors.probs.to(self.device), p_y_uc.to(self.device),
                                  p_c_xu.to(self.device), device=self.device)

            rmse_q_y_x = RMSE(sample_q_y_x[:, 0].to(self.device), q_y_x[:, 0].to(self.device))

        return rmse_q_y_x
    #
    # def fit_all_obs(self, observations):
    #     '''
    #     learn parameters of recognition networks from observations
    #     '''
    #     # stop using batches and just do gradient decent
    #
    #     with torch.no_grad():
    #         num_obs = len(observations['X'])
    #
    #         # Update prior
    #         self.update_prior(num_obs=num_obs)
    #
    #         # Factors distributions are the output of neural network
    #         self.update_factors(observations)
    #
    #         # Update unnormalized probabilities self.generated_factors according to observations (p(c|u,x) and p(y|c,u))
    #         self.update_generated_factors(observations)
    #
    #         # E-Step
    #         self.update_variational()
    #
    #         # Loss
    #         loss = self.get_loss().detach().numpy()
    #     return loss

    def update_variational(self, target=False, observations=None, cond_var=None, entropy_tmp=1.0, prior_tmp=1.0):
        '''
        compute variational q distribution using factors in self.factors
        '''

        # Current Model
        if target:
            if ((observations is None) or (cond_var is None)):
                raise Exception('if target data, must pass observations and cond_var to update_variational')
            prior = Categorical(logits=self.logits_prior[observations[cond_var].to(self.device)].detach())
            factors = self.target_factors
            generated_factors = self.target_generated_factors
            denominators = self.target_denominators
        else:
            prior = self.latent_prior
            factors = self.factors
            generated_factors = self.generated_factors
            denominators = self.denominators

        # Approximate marginals
        log_denominators = sum([denom.logits for denom in denominators])

        # Prior
        log_prior = prior.logits
        # Factors
        log_factor = sum([f_theta_i.logits for f_theta_i in factors])
        # Generated Factors
        log_gen_factor = sum([torch.log(f_i) for f_i in generated_factors])

        # log-prob of the variational
        logits_posterior = (prior_tmp * log_prior + log_gen_factor + log_factor - log_denominators) / (
                    entropy_tmp * prior_tmp)

        if target:
            self.target_q_variational = Categorical(logits=logits_posterior.detach())
        else:
            self.q_variational = Categorical(logits=logits_posterior.detach())

        if torch.sum(torch.isinf(self.q_variational.probs)) > 0:
            raise Exception('got inf in update variational', logits_posterior[torch.isinf(self.q_variational.probs)])
        if torch.sum(torch.isnan(self.q_variational.probs)) > 0:
            raise Exception('got nan in update variational', logits_posterior[torch.isnan(self.q_variational.probs)])
        if torch.sum(torch.isnan(torch.exp(self.q_variational.probs))) > 0:
            raise Exception('got nan in update variational',
                            logits_posterior[torch.isnan(torch.exp(self.q_variational.probs))])
        if torch.sum(torch.sum(torch.exp(self.q_variational.probs), dim=-1) == 0.0) > 0:
            raise Exception('got zero in update variational',
                            logits_posterior[torch.sum(torch.exp(self.q_variational.probs), dim=-1) == 0.0])

    # def update_variational_sudo_gradient(self, target=False, observations=None, cond_var=None, entropy_tmp=1.0,
    #                                      prior_tmp=1.0):
    #     '''
    #     compute variational q distribution using factors in self.factors
    #     '''
    #     learning_rate = self.fit_params['variational_learning_rate']
    #
    #     # Current Model
    #     if target:
    #         if ((observations is None) or (cond_var is None)):
    #             raise Exception('if target data, must pass observations and cond_var to update_variational')
    #         prior = Categorical(logits=self.logits_prior[observations[cond_var].to(self.device)].detach())
    #         factors = self.target_factors
    #         generated_factors = self.target_generated_factors
    #         denominators = self.target_denominators
    #     else:
    #         prior = self.latent_prior
    #         factors = self.factors
    #         generated_factors = self.generated_factors
    #         denominators = self.denominators
    #
    #     # Approximate marginals
    #     log_denominators = sum([denom.logits.to(self.device) for denom in denominators])
    #
    #     # Prior
    #     log_prior = prior.logits.to(self.device)
    #     # Factors
    #     log_factor = sum([f_theta_i.logits.to(self.device) for f_theta_i in factors])
    #     # Generated Factors
    #     log_gen_factor = sum([torch.log(f_i).to(self.device) for f_i in generated_factors])
    #
    #     # log-prob of the variational
    #     logits_posterior = (prior_tmp * log_prior + log_gen_factor + log_factor - log_denominators) / (
    #                 entropy_tmp * prior_tmp)
    #
    #     if target:
    #         self.target_q_variational = Categorical(logits=logits_posterior.detach())
    #     else:
    #         optimal_q_var = Categorical(logits=logits_posterior.detach().to(self.device))
    #         if optimal_q_var.logits.shape[0] == self.q_variational.logits.shape[0]:
    #             grad_logits = learning_rate * (optimal_q_var.logits.to(self.device) - self.q_variational.logits.to(
    #                 self.device)) + self.q_variational.logits.to(self.device)
    #             self.q_variational = Categorical(logits=grad_logits.detach())
    #         else:
    #             print('skipping sudo gradient and just setting variational distribution to optimal')
    #             self.q_variational = optimal_q_var
    #
    #     if torch.sum(torch.isinf(self.q_variational.probs)) > 0:
    #         raise Exception('got inf in update variational', logits_posterior[torch.isinf(self.q_variational.probs)])
    #     if torch.sum(torch.isnan(self.q_variational.probs)) > 0:
    #         raise Exception('got nan in update variational', logits_posterior[torch.isnan(self.q_variational.probs)])
    #     if torch.sum(torch.isnan(torch.exp(self.q_variational.probs))) > 0:
    #         raise Exception('got nan in update variational',
    #                         logits_posterior[torch.isnan(torch.exp(self.q_variational.probs))])
    #     if torch.sum(torch.sum(torch.exp(self.q_variational.probs), dim=-1) == 0.0) > 0:
    #         raise Exception('got zero in update variational',
    #                         logits_posterior[torch.sum(torch.exp(self.q_variational.probs), dim=-1) == 0.0])

    def update_target_variational_posteriors(self):
        '''
        compute variational q distribution for Q(U|X)
        '''

        # target prior
        Q_prior = self.target_latent_prior
        factors = self.target_factors
        denominators = self.target_denominators

        # Prior
        Q_log_prior = Q_prior.logits.detach().to(self.device)
        # P_log_prior = P_prior.logits

        # Denominator terms
        log_denominators = sum([torch.log(denom.probs.to(self.device)).detach() for denom in denominators])

        # Factors
        log_factor = sum([f_theta_i.logits.detach() for f_theta_i in factors]).to(self.device)

        # log-prob of the variational
        logits_posterior = Q_log_prior + log_factor - log_denominators

        self.Q_var_posteriors = Categorical(logits=logits_posterior)

    def update_factors(self, observations, target=False, obs_names=None):
        '''
        compute factor distributions using recognition networks
        '''
        if obs_names is None:
            obs_names = self.obs_names
        num_factors = len(obs_names)
        if type(observations) is DataLoader:
            num_obs = self.num_obs
        else:
            num_obs = len(observations['X'])

        factors_output_tot = [None] * num_factors
        log_denominators = [None] * num_factors
        log_denom_dict = {}
        for indf, factor_name in enumerate(obs_names):
            obs_dim = self.obs_dimensions[factor_name]
            ind_recog = self.recog_network_indices[factor_name]
            # Recognition network
            net = self.recognition_network[ind_recog]
            # Factors NN Output
            if type(observations) is DataLoader:
                factors_output_tot[indf] = batch_net(net, observations, factor_name, None, obs_dim, self.latent_dim,
                                                     self.batch_size, num_obs, self.device)
            else:
                factors_output_tot[indf] = net(observations[factor_name].to(self.device))

            # Update log denominator
            if target:
                log_denominators[indf] = self.log_denom_dict[factor_name].unsqueeze(dim=0).repeat(num_obs, 1)
            else:
                factor_i = Categorical(logits=factors_output_tot[indf])
                log_denom_sum = torch.log(torch.mean(factor_i.probs, 0))
                log_denominators[indf] = log_denom_sum.unsqueeze(dim=0).repeat(num_obs, 1)
                log_denom_dict[factor_name] = log_denom_sum

        # Factors Distribution
        if target:
            self.target_factors = [Categorical(logits=logits.detach()) for logits in factors_output_tot]
            self.target_denominators = [Categorical(logits=log_i.detach()) for log_i in log_denominators]
        else:
            self.factors = [Categorical(logits=logits) for logits in factors_output_tot]
            self.denominators = [Categorical(logits=log_i) for log_i in log_denominators]
            self.log_denom_dict = log_denom_dict


    def update_generated_factors(self, observations, target=False, obs_generated=None):
        '''
        compute generated factor unnormalized probabilities
        '''
        if obs_generated is None:
            obs_generated = self.obs_generated
        num_factors = len(obs_generated)
        if type(observations) is DataLoader:
            num_obs = self.num_obs
        else:
            num_obs = len(observations['X'])

        factors_output_tot = [None] * num_factors
        for indf, factor_name in enumerate(obs_generated):
            obs_dim = self.obs_dimensions[factor_name]
            cond_name = self.obs_conditioned[factor_name]
            ind_gen = self.gen_network_indices[factor_name]
            # Recognition network
            net = self.generation_networks[ind_gen]
            # print('update gen',cond_name, factor_name)
            if type(observations) is DataLoader:
                if target:
                    factors_output_tot[indf] = batch_net(net, observations, None, cond_name, obs_dim,
                                                         self.latent_dim, self.batch_size, num_obs, self.device)
                else:
                    factors_output_tot[indf] = batch_net(net, observations, factor_name, cond_name, obs_dim,
                                                         self.latent_dim, self.batch_size, num_obs, self.device)

            else:
                if target:
                    factor_obs = None
                else:
                    factor_obs = observations[factor_name].to(self.device)

                if cond_name == None:
                    # factor has no conditional
                    # Factors NN Output
                    factors_output_tot[indf] = net(factor_obs)
                else:
                    cond_obs = observations[cond_name].to(self.device)

                    # Factors NN Output
                    factors_output_tot[indf] = net(factor_obs, cond_obs)

        # Factors Distribution (THESE ARE PROBS OF UNNORMALIZED DISTRIBUTION)
        if target:
            return factors_output_tot
        else:
            self.generated_factors = factors_output_tot

        for indf, factor_name in enumerate(obs_generated):
            if torch.sum(torch.isnan(self.generated_factors[indf])) > 0:
                raise Exception('got inf in update generated factors ', factor_name)

    def update_prior(self, target=False, num_obs=None):
        '''
        update the prior over the discrete latent U after self.logits_prior is changed
        '''
        if torch.sum(torch.isinf(self.logits_prior)) > 0:
            raise Exception('got inf in update prior ')
        if num_obs is None:
            num_obs = self.num_obs
        if target:
            self.target_latent_prior = Categorical(logits=self.target_logits_prior.unsqueeze(0).repeat(num_obs, 1))
        else:
            self.latent_prior = Categorical(logits=self.logits_prior.unsqueeze(0).repeat(num_obs, 1))

        if torch.sum(torch.isinf(self.latent_prior.probs)) > 0:
            raise Exception('got inf in update prior ')

    def init_prior(self):
        '''
        initialize the prior over the discrete latent U
        '''
        if self.true_init is None:
            self.logits_prior = nn.Parameter(torch.log(torch.ones(self.latent_dim) / self.latent_dim).to(self.device))
        else:
            self.logits_prior = nn.Parameter(torch.log(self.true_init['true_P_prior']).to(self.device))
        self.latent_prior = Categorical(logits=self.logits_prior.unsqueeze(0).repeat(self.num_obs, 1))

    def init_target_prior(self):
        '''
        initialize the prior Q(U)
        '''
        # new parameter for new prior
        self.target_logits_prior = nn.Parameter(
            torch.log(torch.ones(self.latent_dim) / self.latent_dim).to(self.device))

    def init_variational(self, target=False, num_obs=None):
        '''
        initialize the variational distribution
        '''
        if num_obs is None:
            num_obs = self.num_obs
        if target:
            logits_variational = torch.log(torch.ones(num_obs, self.latent_dim) / self.latent_dim)
            self.target_q_variational = Categorical(logits=logits_variational)
        else:
            logits_variational = torch.log(torch.ones(num_obs, self.latent_dim) / self.latent_dim)
            self.q_variational = Categorical(logits=logits_variational)

    def init_recognition_network(self, dropout):
        '''
        initialize recognition network with separate networks for conditional factors
        '''
        recognition_function = ()
        self.recog_network_indices = {}
        cur_ind = 0
        for ind_f, fact_name in enumerate(self.obs_names):
            cur_dim_obs = self.obs_dimensions[fact_name]
            if self.obs_source[fact_name] == 'MNIST':
                if 'recog_net' in self.fit_params:
                    cur_recognition_function = recognition_net(self.latent_dim, dropout=dropout, data_source='MNIST',
                                                               recog_net=self.fit_params['recog_net']).to(self.device)
                else:
                    cur_recognition_function = recognition_net(self.latent_dim, dropout=dropout,
                                                               data_source='MNIST').to(self.device)
            elif self.obs_source[fact_name] == 'discrete':
                cur_recognition_function = recog_matrix(self.latent_dim, cur_dim_obs).to(self.device)

            elif self.obs_source[fact_name] == 'cifar10':
                if 'recog_net' in self.fit_params:
                    cur_recognition_function = recognition_net(self.latent_dim, dropout=dropout, data_source='cifar10',
                                                               recog_net=self.fit_params['recog_net']).to(self.device)
                else:
                    cur_recognition_function = recognition_net(self.latent_dim, dropout=dropout,
                                                               data_source='cifar10').to(self.device)
            elif self.obs_source[fact_name] == 'vector':
                cur_recognition_function = recognition_net(self.latent_dim,
                                                           input_dim=self.fit_params['observation_dims'][fact_name],
                                                           data_source='vector',
                                                           recog_net=self.fit_params['recog_net']).to(self.device)
            else:
                print(fact_name)
                raise Exception('observation source not specified')
            recognition_function += (cur_recognition_function,)
            self.recog_network_indices[fact_name] = cur_ind
            cur_ind += 1

        # this just makes a modulelist so the parameters are saved correctly
        self.recognition_network = nn.ModuleList(recognition_function)

    def init_generated_probs(self):
        '''
        initialize generated observations p(c|u,x) and p(y|c,u)
        '''
        generation_functions = ()
        self.gen_network_indices = {}
        cur_ind = 0
        for ind_f, fact_name in enumerate(self.obs_generated):
            cur_dim_obs = self.obs_dimensions[fact_name]
            cond_name = self.obs_conditioned[fact_name]
            cond_dim = self.obs_dimensions[cond_name]
            if self.obs_source[cond_name] == 'MNIST':
                cur_generation_function = generated_net(cur_dim_obs, self.latent_dim, data_source='MNIST').to(
                    self.device)
            elif self.obs_source[cond_name] == 'discrete':
                cur_generation_function = generated_matrix(cur_dim_obs, cond_dim, self.latent_dim).to(self.device)
            elif self.obs_source[cond_name] == 'cifar10':
                cur_generation_function = generated_net(cur_dim_obs, self.latent_dim, data_source='cifar10').to(
                    self.device)
            elif self.obs_source[cond_name] == 'vector':
                cur_generation_function = generated_net(cur_dim_obs, self.latent_dim, data_source='vector',
                                                        input_dim=self.fit_params['observation_dims'][cond_name]).to(
                    self.device)
            else:
                print(fact_name)
                raise Exception('observation source not specified')
            generation_functions += (cur_generation_function,)
            self.gen_network_indices[fact_name] = cur_ind
            cur_ind += 1

        # this just makes a modulelist so the parameters are saved correctly
        self.generation_networks = nn.ModuleList(generation_functions)

    def get_posteriors(self, observations, cond_obs, target=False):
        '''
        get variational posteriors q(U|cond_obs)
        '''
        with torch.no_grad():

            # Factors distributions are the output of neural network
            self.update_factors(observations, obs_names=cond_obs, target=target)

            # E-Step
            self.update_variational(target=target)

        if target:
            return self.target_q_variational.probs
        else:
            return self.q_variational.probs

    def get_target_prior_loss(self, num_obs):
        '''
        calculate free energy for target Q loss updating
        '''

        # <log q>_q
        variational_entropy = - categorical_cross_entropy(self.Q_var_posteriors, self.Q_var_posteriors,
                                                          device=self.device)

        # <log Q(U)>_q
        Q_prior_xentropy = categorical_cross_entropy(self.Q_var_posteriors, self.target_latent_prior,
                                                     device=self.device)

        # <log f(u)>_q
        denom_xentropy = sum([categorical_cross_entropy(self.Q_var_posteriors, denom, device=self.device) for denom in
                              self.target_denominators])

        # <log f(U|.)>_q
        factor_xentropy = sum(
            [categorical_cross_entropy(self.Q_var_posteriors, f_theta_i, device=self.device) for f_theta_i in
             self.target_factors])

        # Free Energy / ELBO
        free_energy = \
            torch.sum(variational_entropy + Q_prior_xentropy + factor_xentropy - denom_xentropy)

        return - free_energy / float(num_obs)



    def learn_prior_only(self, observations):
        '''
        learn only prior parameters and freeze recognition networks
        '''
        if self.target_logits_prior is None:
            self.init_target_prior()

        # Fit parameters
        learn_epochs = self.fit_params['epochs']
        # num_obs = len(observations['X'])
        if type(observations) is DataLoader:
            num_obs = len(observations.dataset)
        else:
            num_obs = observations['num_obs']

        for param in self.recognition_network.parameters():
            if param.requires_grad:
                param.requires_grad = False
        for param in self.generation_networks.parameters():
            if param.requires_grad:
                param.requires_grad = False
        self.logits_prior.requires_grad = False


        # Network Optimizer  Parameters
        if ('target_prior_learn_rate' in self.fit_params):
            learning_rate = self.fit_params['target_prior_learn_rate']
        else:
            learning_rate = self.fit_params['prior_learn_rate']
        optimizer_Qprior = torch.optim.Adam([self.target_logits_prior], lr=learning_rate)

        # Init Loss
        loss_tot = np.zeros(learn_epochs)

        # for timing
        runtime = np.zeros(learn_epochs)
        startTime = time.time()

        # Update factors with target observations
        self.update_factors(observations, obs_names=['X'], target=True)

        # Update prior with new prior
        self.init_variational(target=True, num_obs=num_obs)

        # Clear gradients for this training step
        optimizer_Qprior.zero_grad()

        with torch.autograd.set_detect_anomaly(True):
            print('num epochs learning target prior', learn_epochs)
            for epoch in range(learn_epochs):
                # Update prior after learning
                self.update_prior(target=True, num_obs=num_obs)

                with torch.no_grad():
                    # E-Step
                    self.update_target_variational_posteriors()

                # Loss and Backprop
                loss = self.get_target_prior_loss(num_obs)
                loss.backward()

                # M-Step apply gradients
                optimizer_Qprior.step()

                # Clear gradients for this training step
                optimizer_Qprior.zero_grad()

                loss_tot[epoch] = -loss.detach()

                # keep track of run time
                curTime = time.time()
                runtime[epoch] = curTime - startTime

        for param in self.recognition_network.parameters():
            param.requires_grad = True
        for param in self.generation_networks.parameters():
            param.requires_grad = True
        self.logits_prior.requires_grad = True

        return loss_tot, runtime

    def get_generated_probs(self, observations=None):
        '''
        return P(Y|C,U) and P(C|X,U)
        '''
        if observations is None:
            gen_probs = {}
            for indf, factor_name in enumerate(self.obs_generated):
                ind_gen = self.gen_network_indices[factor_name]
                prob_matrix = self.generation_networks[ind_gen].get_prob_matrix()
                gen_probs[factor_name] = prob_matrix
            return gen_probs
        else:
            with torch.no_grad():
                # get P(C|U,X) for every x in target
                gen_probs = {}
                gen_probs['C'] = self.update_generated_factors(observations, target=True, obs_generated=['C'])[0]

                # get P(Y|U,C) matrix
                ind_gen = self.gen_network_indices['Y']
                prob_matrix = self.generation_networks[ind_gen].get_prob_matrix()
                gen_probs['Y'] = prob_matrix

                return gen_probs

    # def get_joint(self, source_obs):
    #     '''
    #     get joint probability P(U,Y,C,X,W)
    #     '''
    #
    #     obs = torch.eye(self.obs_dimensions['X'])[np.arange(self.obs_dimensions['X'])]
    #     ind_recog = self.recog_network_indices['X']
    #     net = self.recognition_network[ind_recog]
    #     f_u_x = torch.permute(torch.exp(net(obs)), (1, 0))  # U x X
    #     # print('f_u_x', f_u_x)
    #
    #     obs = torch.eye(self.obs_dimensions['W'])[np.arange(self.obs_dimensions['W'])]
    #     ind_recog = self.recog_network_indices['W']
    #     net = self.recognition_network[ind_recog]
    #     f_u_w = torch.permute(torch.exp(net(obs)), (1, 0))  # U x W
    #     # print('f_u_w', f_u_w)
    #
    #     ind_gen = self.gen_network_indices['C']
    #     for name, param in self.generation_networks[ind_gen].named_parameters():
    #         if param.requires_grad:
    #             prob_mat = torch.exp(param.data.clone().detach())
    #             p_c_xu = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
    #     # print('p_c_xu',p_c_xu)
    #
    #     ind_gen = self.gen_network_indices['Y']
    #     for name, param in self.generation_networks[ind_gen].named_parameters():
    #         if param.requires_grad:
    #             prob_mat = torch.exp(param.data.clone().detach())
    #             p_y_cu = prob_mat / torch.sum(prob_mat, dim=0).unsqueeze(0)
    #     # print('p_y_cu',p_y_cu)
    #
    #     p_u = self.latent_prior.probs[0]
    #     # print('p_u',p_u)
    #     denom_0 = self.denominators[0].probs[0]
    #     denom_1 = self.denominators[1].probs[1]
    #     # print('denom 0',denom_0)
    #     # print('denom 1',denom_1)
    #
    #     p_x, n = observed_conditional(['X'], [], source_obs, self.obs_dimensions)
    #     p_w, n = observed_conditional(['W'], [], source_obs, self.obs_dimensions)
    #     # print('p_x',p_x)
    #     # print('p_w',p_w)
    #
    #     learn_joint = torch.einsum('x,w,ux,uw,u,u,cxu,ycu,u->uycxw', p_x, p_w, f_u_x, f_u_w, 1.0 / denom_0,
    #                                1.0 / denom_1, p_c_xu, p_y_cu, p_u)
    #     return learn_joint.detach()

    # def get_joint_MNIST(self, source_obs):
    #     '''
    #     get joint probability P(U,Y,C,X,W) by averaging factors for MNIST values
    #     '''
    #     permutation = torch.arange(self.latent_dim).to(self.device)
    #     f_u_w = posterior_averages(source_obs, self.factors[0].probs, ['W'], self.obs_dimensions, permutation).T.to(
    #         self.device)
    #     f_u_x = posterior_averages(source_obs, self.factors[1].probs, ['X'], self.obs_dimensions, permutation).T.to(
    #         self.device)
    #     p_c_xu = posterior_averages(source_obs, self.generated_factors[0], ['C', 'X'], self.obs_dimensions,
    #                                 permutation).to(self.device)
    #     p_y_cu = posterior_averages(source_obs, self.generated_factors[1], ['Y', 'C'], self.obs_dimensions,
    #                                 permutation).to(self.device)
    #
    #     p_u = self.latent_prior.probs[0].to(self.device)
    #     # print('p_u',p_u)
    #     denom_0 = self.denominators[0].probs[0].to(self.device)
    #     denom_1 = self.denominators[1].probs[1].to(self.device)
    #     # print('denom 0',denom_0)
    #     # print('denom 1',denom_1)
    #
    #     p_x, n = observed_conditional(['X'], [], source_obs, self.obs_dimensions)
    #     p_w, n = observed_conditional(['W'], [], source_obs, self.obs_dimensions)
    #     # print('p_x',p_x)
    #     # print('p_w',p_w)
    #     print('shapes', p_x.shape, p_w.shape, f_u_x.shape, f_u_w.shape, denom_0.shape, denom_1.shape, p_c_xu.shape,
    #           p_y_cu.shape, p_u.shape)
    #
    #     learn_joint = torch.einsum('x,w,ux,uw,u,u,cxu,ycu,u->uycxw', p_x.to(self.device), p_w.to(self.device), f_u_x,
    #                                f_u_w, 1.0 / denom_0, 1.0 / denom_1, p_c_xu, p_y_cu, p_u)
    #     return learn_joint.detach()

