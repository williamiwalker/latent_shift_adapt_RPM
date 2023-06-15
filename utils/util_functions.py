import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from scipy.optimize import linear_sum_assignment
import re, os, math

import torchvision.datasets  # import CIFAR100
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import pandas as pd


def categorical_cross_entropy(p1: Categorical, p2: Categorical, device='cpu'):
    # Cross Entropy Between 2 categorical distributions
    min_real = torch.finfo(p2.logits.dtype).min
    logits_p2 = torch.clamp(p2.logits.to(device), min=min_real)
    return (p1.probs.to(device) * logits_p2).sum(-1)


def unnormalized_categorical_cross_entropy(p1: Categorical, probs_p2, device='cpu'):
    # Cross Entropy Between 2 categorical distributions
    log_p2 = torch.log(probs_p2.to(device) + 1e-20)
    return (p1.probs.to(device) * log_p2).sum(-1)


def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))




def sigmoid_mat(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)[np.newaxis, :]


def small_discrete_example_alt(num_obs, device, all_indices=False, random=False, dimensions=(3, 2, 3, 2, 3)):
    '''
    create observations
    :param random: if True, pick random values for all probabilities. if False, use the values from the paper
    '''
    q_num_obs = num_obs
    k_u, k_x, k_c, k_y, k_w = dimensions
    if random:
        k_u, k_x, k_c, k_y, k_w = dimensions
    else:
        # deterministic u->x and u->w. dm_x=2
        k_u, k_x, k_c, k_y, k_w = dimensions
        p_u = sigmoid(np.array([1.0, 0.1, 0.1]))
        q_u = sigmoid(np.array([0.1, 0.1, 1.0]))
        M_w_u = np.array([[1e2, -1e20, -1e20], [-1e20, 1e2, -1e20], [-1e20, -1e20, 1e2]])
        M_x_u = np.array([[1e2, 1e2, -1e20], [1e2, -1e20, 1e2]])
        M_c_x = np.array([[5.0, 0.5], [5.0, 5.0], [0.5, 5.0]])
        M_c_u = np.array([[5.0, 5.0, 0.5], [5.0, 0.5, 5.0], [0.5, 5.0, 5.0]])
        M_y_c = np.array([[5.0, 5.0, 0.5], [0.5, 5.0, 5.0]])
        M_y_u = M_y_c

    u_obs = np.zeros(num_obs, dtype=int)
    x_obs = np.zeros(num_obs, dtype=int)
    y_obs = np.zeros(num_obs, dtype=int)
    w_obs = np.zeros(num_obs, dtype=int)
    c_obs = np.zeros(num_obs, dtype=int)

    q_u_obs = np.zeros(q_num_obs, dtype=int)
    q_x_obs = np.zeros(q_num_obs, dtype=int)
    q_c_obs = np.zeros(q_num_obs, dtype=int)
    q_y_obs = np.zeros(q_num_obs, dtype=int)

    u_obs = np.random.choice(k_u, size=num_obs, p=p_u)
    q_u_obs = np.random.choice(k_u, size=q_num_obs, p=q_u)
    for indo in range(num_obs):
        w_obs[indo] = np.random.choice(k_w, size=1, p=sigmoid(M_w_u[:, u_obs[indo]]))
        x_obs[indo] = np.random.choice(k_x, size=1, p=sigmoid(M_x_u[:, u_obs[indo]]))
        c_obs[indo] = np.random.choice(k_c, size=1, p=sigmoid(M_c_x[:, x_obs[indo]] + M_c_u[:, u_obs[indo]]))
        y_obs[indo] = np.random.choice(k_y, size=1, p=sigmoid(M_y_c[:, c_obs[indo]] + M_y_u[:, u_obs[indo]]))
    for indo in range(q_num_obs):
        q_x_obs[indo] = np.random.choice(k_x, size=1, p=sigmoid(M_x_u[:, q_u_obs[indo]]))
        q_c_obs[indo] = np.random.choice(k_c, size=1, p=sigmoid(M_c_x[:, q_x_obs[indo]] + M_c_u[:, q_u_obs[indo]]))
        q_y_obs[indo] = np.random.choice(k_y, size=1, p=sigmoid(M_y_c[:, q_c_obs[indo]] + M_y_u[:, q_u_obs[indo]]))

    if all_indices:
        source_obs = {'U': torch.from_numpy(u_obs).to(device), 'X': torch.from_numpy(x_obs).to(device),
                      'Y': torch.from_numpy(y_obs).to(device), 'W': torch.from_numpy(w_obs).to(device),
                      'C': torch.from_numpy(c_obs).to(device), 'num_obs': num_obs}
        target_obs = {'U': torch.from_numpy(q_u_obs).to(device), 'X': torch.from_numpy(q_x_obs).to(device),
                      'Y': torch.from_numpy(q_y_obs).to(device), 'num_obs': q_num_obs}
    else:
        source_obs = {'U': torch.from_numpy(u_obs).to(device), 'X': torch.eye(k_x).to(device)[x_obs],
                      'Y': torch.eye(k_y).to(device)[y_obs], 'W': torch.eye(k_w).to(device)[w_obs],
                      'C': torch.from_numpy(c_obs).to(device), 'num_obs': num_obs}
        target_obs = {'U': torch.from_numpy(q_u_obs).to(device), 'X': torch.eye(k_x).to(device)[q_x_obs],
                      'Y': torch.eye(k_y).to(device)[q_y_obs], 'num_obs': q_num_obs}

    P_W_U = sigmoid_mat(M_w_u)
    P_X_U = sigmoid_mat(M_x_u)
    P_C_XU = sigmoid_mat(M_c_x[:, :, np.newaxis] + M_c_u[:, np.newaxis, :])
    P_Y_CU = sigmoid_mat(M_y_c[:, :, np.newaxis] + M_y_u[:, np.newaxis, :])
    full_joint = np.einsum('wu,xu,cxu,ycu,u->uycxw', P_W_U, P_X_U, P_C_XU, P_Y_CU, p_u)
    target_full_joint = np.einsum('wu,xu,cxu,ycu,u->uyx', P_W_U, P_X_U, P_C_XU, P_Y_CU, q_u)

    return source_obs, target_obs, full_joint, target_full_joint



def MNIST_batches_with_replacement(source_obs, target_obs, batch_size, device, sample_size=5000, shuffle=False,
                                   dimensions=(3, 2, 3, 2, 3)):
    '''
    make a dataset with shuffle on
    '''

    # set seed so all runs use same dataset
    np.random.seed(seed=42)

    num_diff_samples = sample_size

    num_obs = len(source_obs['Y'])
    q_num_obs = len(target_obs['Y'])
    k_u, k_x, k_c, k_y, k_w = dimensions

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    train_data = torchvision.datasets.MNIST(download=True, train=True, root="./data", transform=transform)

    # replace x_obs with MNIST images
    labels = train_data.train_labels
    # save all images
    X_image_obs = torch.zeros(num_obs, 1, 28, 28)
    # keep track of which images were used in dataset
    used_x_indices = torch.zeros(k_x, dtype=int)
    x_obs = source_obs['X']
    for ind_x in range(k_x):
        # make observations indices from one-hot
        one_hot = torch.eye(k_x)[ind_x]
        # mask of which observations have ind_x value
        obs_indices = torch.all(x_obs == one_hot, dim=1)
        # get indices of MNIST digits with correct labels
        label_indices = (labels == ind_x).nonzero(as_tuple=True)[0]
        cur_images = torch.zeros(torch.sum(obs_indices), 1, 28, 28)

        # fill cur_images
        for ind_c in range(torch.sum(obs_indices)):
            cur_images[ind_c] = train_data[label_indices[np.random.randint(num_diff_samples)]][0]

        # make dataset with cur_images
        X_image_obs[obs_indices] = cur_images

    # replace w_obs with MNIST images
    W_image_obs = torch.zeros(num_obs, 1, 28, 28)
    w_obs = source_obs['W']
    for ind_w in range(k_w):
        # make observations indices from one-hot
        one_hot = torch.eye(k_w)[ind_w]
        obs_indices = torch.all(w_obs == one_hot, dim=1)
        # print('labels',torch.sum(labels == int(k_x + ind_w)))
        label_indices = (labels == int(k_x + ind_w)).nonzero(as_tuple=True)[0]
        # print('label indices',label_indices)
        cur_images = torch.zeros(torch.sum(obs_indices), 1, 28, 28)

        # fill cur_images
        for ind_c in range(torch.sum(obs_indices)):
            cur_images[ind_c] = train_data[label_indices[np.random.randint(num_diff_samples)]][0]

        # make dataset with cur_images
        W_image_obs[obs_indices] = cur_images

    # replace q_xx_obs with MNIST images
    QX_image_obs = torch.zeros(q_num_obs, 1, 28, 28)
    q_x_obs = target_obs['X']
    for ind_x in range(k_x):
        # make observations indices from one-hot
        one_hot = torch.eye(k_x)[ind_x]
        # mask of which observations have ind_x value
        obs_indices = torch.all(q_x_obs == one_hot, dim=1)
        label_indices = (labels == ind_x).nonzero(as_tuple=True)[0]
        cur_images = torch.zeros(torch.sum(obs_indices), 1, 28, 28)

        # fill cur_images
        for ind_c in range(torch.sum(obs_indices)):
            cur_images[ind_c] = train_data[label_indices[np.random.randint(num_diff_samples)]][0]

        # make dataset with cur_images
        QX_image_obs[obs_indices] = cur_images

    source_dset = DiscreteMNISTDataset(source_obs['U'], source_obs['Y'], c=source_obs['C'],
                                       x_images=X_image_obs, w_images=W_image_obs, true_x=source_obs['X'],
                                       true_w=source_obs['W'])
    source_dloader = DataLoader(source_dset, batch_size, num_workers=1, pin_memory=True, shuffle=shuffle)
    target_dset = DiscreteMNISTDataset(target_obs['U'], target_obs['Y'], x_images=QX_image_obs)
    target_dloader = DataLoader(target_dset, batch_size, num_workers=1, pin_memory=True, shuffle=shuffle)

    return source_dloader, target_dloader


def cifar_batches_with_replacement(source_obs, target_obs, batch_size, device, sample_size=5000, shuffle=False,
                                   dimensions=(3, 2, 3, 2, 3)):
    '''
    make a dataset with shuffle on
    '''

    # set seed so all runs use same dataset
    np.random.seed(seed=42)

    num_diff_samples = sample_size

    num_obs = len(source_obs['Y'])
    q_num_obs = len(target_obs['Y'])
    k_u, k_x, k_c, k_y, k_w = dimensions

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = torchvision.datasets.CIFAR10(download=True, train=True, root="./data", transform=transform)

    # replace x_obs with MNIST images
    labels = torch.tensor(train_data.targets)
    # save all images
    X_image_obs = torch.zeros(num_obs, 3, 32, 32)
    # keep track of which images were used in dataset
    used_x_indices = torch.zeros(k_x, dtype=int)
    x_obs = source_obs['X']
    for ind_x in range(k_x):
        # make observations indices from one-hot
        one_hot = torch.eye(k_x)[ind_x]
        # mask of which observations have ind_x value
        obs_indices = torch.all(x_obs == one_hot, dim=1)
        # get indices of MNIST digits with correct labels
        label_indices = (labels == ind_x).nonzero(as_tuple=True)[0]
        cur_images = torch.zeros(torch.sum(obs_indices), 3, 32, 32)

        # fill cur_images
        for ind_c in range(torch.sum(obs_indices)):
            cur_images[ind_c] = train_data[label_indices[np.random.randint(num_diff_samples)]][0]

        # make dataset with cur_images
        X_image_obs[obs_indices] = cur_images

    # replace w_obs with MNIST images
    W_image_obs = torch.zeros(num_obs, 3, 32, 32)
    w_obs = source_obs['W']
    for ind_w in range(k_w):
        # make observations indices from one-hot
        one_hot = torch.eye(k_w)[ind_w]
        obs_indices = torch.all(w_obs == one_hot, dim=1)
        # print('labels',torch.sum(labels == int(k_x + ind_w)))
        label_indices = (labels == int(k_x + ind_w)).nonzero(as_tuple=True)[0]
        # print('label indices',label_indices)
        cur_images = torch.zeros(torch.sum(obs_indices), 3, 32, 32)

        # fill cur_images
        for ind_c in range(torch.sum(obs_indices)):
            cur_images[ind_c] = train_data[label_indices[np.random.randint(num_diff_samples)]][0]

        # make dataset with cur_images
        W_image_obs[obs_indices] = cur_images

    # replace q_xx_obs with MNIST images
    QX_image_obs = torch.zeros(q_num_obs, 3, 32, 32)
    q_x_obs = target_obs['X']
    for ind_x in range(k_x):
        # make observations indices from one-hot
        one_hot = torch.eye(k_x)[ind_x]
        # mask of which observations have ind_x value
        obs_indices = torch.all(q_x_obs == one_hot, dim=1)
        label_indices = (labels == ind_x).nonzero(as_tuple=True)[0]
        cur_images = torch.zeros(torch.sum(obs_indices), 3, 32, 32)

        # fill cur_images
        for ind_c in range(torch.sum(obs_indices)):
            cur_images[ind_c] = train_data[label_indices[np.random.randint(num_diff_samples)]][0]

        # make dataset with cur_images
        QX_image_obs[obs_indices] = cur_images

    source_dset = DiscreteMNISTDataset(source_obs['U'], source_obs['Y'], c=source_obs['C'],
                                       x_images=X_image_obs, w_images=W_image_obs, true_x=source_obs['X'],
                                       true_w=source_obs['W'])
    source_dloader = DataLoader(source_dset, batch_size, num_workers=1, pin_memory=True, shuffle=shuffle)
    target_dset = DiscreteMNISTDataset(target_obs['U'], target_obs['Y'], x_images=QX_image_obs)
    target_dloader = DataLoader(target_dset, batch_size, num_workers=1, pin_memory=True, shuffle=shuffle)

    return source_dloader, target_dloader




class MNISTDataset(Dataset):
    def __init__(self, images):
        # self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # label = self.labels[idx]
        image = self.images[idx]
        # sample = {"image": images, "class": label}
        return image


class DiscreteDataset(Dataset):
    def __init__(self, u, y, c, x, w):
        self.u = u
        self.y = y
        self.c = c
        self.x = x
        self.w = w

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {"U": self.u[idx], "Y": self.y[idx], 'C': self.c[idx], 'X': self.x[idx], 'W': self.w[idx]}
        return sample


class DiscreteMNISTDataset(Dataset):
    def __init__(self, u, y, c=None, x_images=None, w_images=None, true_x=None, true_w=None):
        self.u = u
        self.y = y
        self.c = c
        self.x_images = x_images
        self.w_images = w_images
        self.true_x = true_x
        self.true_w = true_w

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.c is None:
            sample = {"U": self.u[idx], "Y": self.y[idx], 'X': self.x_images[idx]}
        else:
            if self.true_x is not None:
                sample = {"U": self.u[idx], "Y": self.y[idx], 'C': self.c[idx], 'X': self.x_images[idx],
                          'W': self.w_images[idx], "true_X": self.true_x[idx], "true_W": self.true_w[idx]}
            else:
                sample = {"U": self.u[idx], "Y": self.y[idx], 'C': self.c[idx], 'X': self.x_images[idx],
                          'W': self.w_images[idx]}
        return sample



def save_checkpoint(model, step, lossTrack, checkpoint_dir):
    '''
    save model
    :param model: RPM_HMM model
    :param step: current epoch during training
    :param lossTrack: history of loss for all previous epochs
    :param checkpoint_dir: save folder
    :return:
    '''
    factor_logits = [factor.logits.clone().detach().cpu().numpy() for factor in model.factors]
    checkpoint_state = {
        "model": model.state_dict(),
        "step": step,
        "loss": lossTrack,
        "q_variational_logits": model.q_variational.logits.clone().detach().cpu().numpy(),
        "latent_prior_logits": model.latent_prior.logits.clone().detach().cpu().numpy(),
        "factor_logits": factor_logits,
        "log_denom_dict": model.log_denom_dict}
    if hasattr(model, 'generated_factors'):
        gen_probs = [factor.clone().detach().cpu().numpy() for factor in model.generated_factors]
        checkpoint_state["generated_factors"] = gen_probs
    if hasattr(model, 'Q_var_posteriors'):
        checkpoint_state["Q_var_posteriors_logits"] = model.Q_var_posteriors.logits.clone().detach().cpu().numpy()
    if hasattr(model, 'denominators'):
        denom_probs = [denom.logits.clone().detach().cpu().numpy() for denom in model.denominators]
        checkpoint_state["denominators_logits"] = denom_probs
    if hasattr(model, 'log_denom_dict'):
        checkpoint_state["log_denom_dict"] = model.log_denom_dict
    if hasattr(model, 'target_factors'):
        logits = [factor.logits.clone().detach().cpu().numpy() for factor in model.target_factors]
        checkpoint_state["target_factors_logits"] = logits
    if hasattr(model, 'target_denominators'):
        logits = [factor.logits.clone().detach().cpu().numpy() for factor in model.target_denominators]
        checkpoint_state["target_denominators_logits"] = logits
    # if hasattr(model, 'q_transition_alpha'):
    #     checkpoint_state["q_transition_alpha"] = model.q_transition_alpha.clone().detach().cpu().numpy()
    # else:
    #     checkpoint_state["transition_matrix"] = model.transition_matrix.clone().detach().cpu().numpy()
    checkpoint_path = checkpoint_dir + 'learned_model.pth'  # / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def load_model(model, model_folder, model_name, device):
    '''
    load model state
    '''
    stateDict = torch.load(model_folder + model_name + '.pth', map_location=device)  # torch.device(device))
    model.load_state_dict(stateDict['model'])
    model.latent_prior = Categorical(logits=model.logits_prior)
    model.q_variational = Categorical(logits=model.target_logits_prior)
    model.factors = [Categorical(logits=torch.tensor(factor, device=device)) for factor in stateDict['factor_logits']]
    model.log_denom_dict = stateDict['log_denom_dict']
    if 'generated_factors' in stateDict.keys():
        model.generated_factors = [torch.tensor(factor, device=device) for factor in stateDict['generated_factors']]
    if 'Q_var_posteriors_logits' in stateDict.keys():
        model.Q_var_posteriors = Categorical(logits=torch.tensor(stateDict['Q_var_posteriors_logits'], device=device))
    if 'denominators_logits' in stateDict.keys():
        model.denominators = [Categorical(logits=torch.tensor(factor, device=device)) for factor in
                              stateDict['denominators_logits']]
    if 'log_denom_dict' in stateDict.keys():
        model.log_denom_dict = stateDict['log_denom_dict']
    if 'target_factors_logits' in stateDict.keys():
        model.target_factors = [Categorical(logits=torch.tensor(factor, device=device)) for factor in
                                stateDict['target_factors_logits']]
    if 'target_denominators_logits' in stateDict.keys():
        model.target_denominators = [Categorical(logits=torch.tensor(factor, device=device)) for factor in
                                     stateDict['target_denominators_logits']]

    print('finished loading model')
    return model



def permute_latent(latent_log_probs, true_latent, latent_dim):
    '''
    Find Best Prediction in case the train labels are permuted
    '''

    # Score Used to find Opt. Permutation
    scor_tot = np.eye(latent_dim)

    for cur_latent in range(latent_dim):
        # Prediction digit = digit_cur
        idx = (true_latent == cur_latent)

        # How many times 'digit_cur' from prediction is mapped to each digit in label.
        scor_tot[cur_latent, :] = latent_log_probs[idx].sum(0)
        print('score', scor_tot[cur_latent, :])

    # Maximise scor_tot using Hungarian Algorithm
    _, perm = linear_sum_assignment(-scor_tot)

    return perm


def recursive_indexing(dims, cur_ind):
    '''
    returns index of cur_ind and recursively calls next cur_ind+=1
    '''
    if cur_ind >= len(dims):
        return [np.array([], dtype=int)]
    else:
        ind_list = []
        for indd in range(dims[cur_ind]):
            rest_list = recursive_indexing(dims, cur_ind + 1)
            ind_list = ind_list + list(
                np.concatenate((np.expand_dims(np.full(len(rest_list), indd, dtype=int), 1), rest_list), axis=1))
        return ind_list


def recursive_assignment(matrix, index, value):
    '''
    assign value to index in matrix
    '''
    if len(index) == 1:
        matrix[index[0]] = value
        return matrix
    else:
        return recursive_assignment(matrix[index[0]], index[1:], value)


def posterior_counts(observations, cond_vars, dims):
    '''
    find count probabilities of latent conditioned on cond_vars
    '''
    sampled_latents = observations['U']
    num_obs = len(sampled_latents)
    cond_dims = [None] * len(cond_vars)
    for indc, cond_name in enumerate(cond_vars):
        cond_dims[indc] = dims[cond_name]

    posteriors = np.zeros(cond_dims + [dims['U']])
    num_samples = np.zeros(cond_dims + [1])

    all_indices = recursive_indexing(cond_dims, 0)
    # print(all_indices)

    for cond_ind in all_indices:
        mask = np.ones(num_obs, dtype=bool)
        for indc, cond_name in enumerate(cond_vars):
            if len(observations[cond_name][0].shape) == 0:
                # print('here')
                mask = mask * (observations[cond_name].numpy() == cond_ind[indc])
                # print('next mask', torch.sum((observations[cond_name] == cond_ind[indc])))
            else:
                one_hot = np.eye(cond_dims[indc])[cond_ind[indc]]
                # print('one hot',one_hot)
                mask = mask * np.all(observations[cond_name].numpy() == one_hot, axis=1)
                # print('new mask ', np.sum(np.all(observations[cond_name].numpy() == one_hot, axis=1)))

        one_hot_latent = np.eye(dims['U'])
        if len(cond_ind) == 2:
            # print('num obs',np.sum(mask), mask.shape, np.sum(one_hot_latent[sampled_latents[mask]],axis=0))
            if np.sum(mask) == 1:
                posteriors[cond_ind[0], cond_ind[1]] = one_hot_latent[sampled_latents[mask]]
            elif np.sum(mask) > 0:
                posteriors[cond_ind[0], cond_ind[1]] = np.sum(one_hot_latent[sampled_latents[mask]], axis=0) / float(
                    np.sum(mask))
            num_samples[cond_ind[0], cond_ind[1]] = np.sum(mask)
        elif len(cond_ind) == 1:
            # print('num obs',np.sum(mask), mask.shape, np.sum(one_hot_latent[sampled_latents[mask]],axis=0))
            if np.sum(mask) == 1:
                posteriors[cond_ind[0]] = one_hot_latent[sampled_latents[mask]]
            elif np.sum(mask) > 0:
                posteriors[cond_ind[0]] = np.sum(one_hot_latent[sampled_latents[mask]], axis=0) / float(
                    np.sum(mask))
            num_samples[cond_ind[0]] = np.sum(mask)

    return posteriors, num_samples


def posterior_averages(observations, posteriors, cond_vars, dims, permute):
    '''
    find count probabilities of latent conditioned on cond_vars
    '''
    num_obs = observations['num_obs']
    cond_dims = [None] * len(cond_vars)
    for indc, cond_name in enumerate(cond_vars):
        cond_dims[indc] = dims[cond_name]

    if len(posteriors.shape) == 2:
        last_dim = posteriors.shape[-1]
        avg_posteriors = torch.zeros(cond_dims + [last_dim])
    elif len(posteriors.shape) == 3:
        last_dims = list(posteriors.shape[1:])
        avg_posteriors = torch.zeros(cond_dims + last_dims)

    all_indices = recursive_indexing(cond_dims, 0)
    # print(all_indices)

    for cond_ind in all_indices:
        mask = torch.ones(num_obs, dtype=bool)
        for indc, cond_name in enumerate(cond_vars):
            # print('cond name',cond_name, len(observations[cond_name][0].shape))
            if len(observations[cond_name][0].shape) == 0:
                # print('here')
                mask = mask * (observations[cond_name] == cond_ind[indc])
                # print('next mask', torch.sum((observations[cond_name] == cond_ind[indc])))
                # if torch.sum(mask) == 0.0:
                #     print('sum is zero ', cond_name)
            else:
                one_hot = torch.eye(cond_dims[indc])[cond_ind[indc]]
                # print('one hot',one_hot)
                mask = mask * torch.all(observations[cond_name] == one_hot, dim=1)
                # print('new mask ', np.sum(np.all(observations[cond_name].numpy() == one_hot, axis=1)))
                # if torch.sum(mask) == 0.0:
                #     print('sum is zero one-hot ', cond_name)
        if torch.sum(mask) > 0:
            if len(cond_vars) == 1:
                avg_posteriors[cond_ind[0]] = (torch.sum(posteriors[mask], dim=0) / torch.sum(mask))[permute]
            elif len(cond_vars) == 2:
                avg_posteriors[cond_ind[0], cond_ind[1]] = (torch.sum(posteriors[mask], dim=0) / torch.sum(mask))[
                    permute]
            elif len(cond_vars) == 3:
                avg_posteriors[cond_ind[0], cond_ind[1], cond_ind[2]] = \
                (torch.sum(posteriors[mask], dim=0) / torch.sum(mask))[permute]
            elif len(cond_vars) == 4:
                avg_posteriors[cond_ind[0], cond_ind[1], cond_ind[2], cond_ind[3]] = \
                (torch.sum(posteriors[mask], dim=0) / torch.sum(mask))[permute]

    return avg_posteriors


def observed_conditional(var_names, cond_vars, observations, dims, nump=False):
    '''
    find count probabilities of latent conditioned on cond_vars
    '''
    num_obs = len(observations[var_names[0]])
    cond_dims = [None] * len(cond_vars)
    var_dims = [None] * len(var_names)
    for indc, cond_name in enumerate(cond_vars):
        if bool(re.search('true_*', cond_name)):
            cond_dims[indc] = dims[cond_name[5:]]
        else:
            cond_dims[indc] = dims[cond_name]
    for indv, var in enumerate(var_names):
        if bool(re.search('true_*', var)):
            var_dims[indv] = dims[var[5:]]
        else:
            var_dims[indv] = dims[var]

    all_dims = var_dims + cond_dims

    probs = torch.zeros(all_dims)
    num_samples = torch.zeros(cond_dims)

    all_indices = recursive_indexing(all_dims, 0)
    # print(all_indices)

    for prob_ind in all_indices:
        mask = torch.ones(num_obs, dtype=bool)
        for indc, cond_name in enumerate(cond_vars):
            # conditioned variable mask
            if len(observations[cond_name][0].shape) == 0:
                mask = mask * (observations[cond_name] == prob_ind[indc + len(var_dims)])
            else:
                one_hot = torch.eye(dims[cond_name])[prob_ind[indc + len(var_dims)]]
                mask = mask * torch.all(observations[cond_name] == one_hot, axis=1)

        mask_var = mask.clone()
        for indv, var in enumerate(var_names):
            # first variable counts
            if len(observations[var][0].shape) == 0:
                mask_var = mask_var * (observations[var] == prob_ind[indv])
            else:
                one_hot = torch.eye(dims[var])[prob_ind[indv]]
                mask_var = mask_var * (torch.all(observations[var] == one_hot, axis=1))

        # print('num obs',torch.sum(mask), mask.shape, torch.sum(one_hot_latent[sampled_latents[mask]],axis=0))
        if torch.sum(mask) > 0:
            matrix = recursive_assignment(probs, prob_ind, float(torch.sum(mask_var)) / float(torch.sum(mask)))
            # print('matrix', matrix, probs)
            # probs[prob_ind[0],prob_ind[1]] = float(torch.sum(mask_var))/float(torch.sum(mask))
        # else:
        #     print('no values here',torch.sum(mask))
        if len(cond_vars) == 0:
            num_samples = torch.sum(mask)
        elif len(cond_vars) == 1:
            num_samples[prob_ind[0 + len(var_dims)]] = torch.sum(mask)
        elif len(cond_vars) == 2:
            num_samples[prob_ind[0 + len(var_dims)], prob_ind[1 + len(var_dims)]] = torch.sum(mask)
        elif len(cond_vars) == 3:
            num_samples[
                prob_ind[0 + len(var_dims)], prob_ind[1 + len(var_dims)], prob_ind[2 + len(var_dims)]] = torch.sum(mask)
        elif len(cond_vars) == 4:
            num_samples[prob_ind[0 + len(var_dims)], prob_ind[1 + len(var_dims)], prob_ind[2 + len(var_dims)], prob_ind[
                3 + len(var_dims)]] = torch.sum(mask)

    if nump:
        return probs.numpy(), num_samples.numpy()
    else:
        return probs, num_samples


def predict_q_y_x(observations, q_u_x, p_y_uc, p_c_ux, device='cpu'):
    '''
    Q(Y|X) \propto \sum_U \sum_C P(Y|U,C) P(C|U,X) Q(U|X)
    '''

    if p_c_ux.shape[0] == p_y_uc.shape[-1]:
        # turn observations from one-hot to index
        x_obs = torch.argmax(observations['X'].to(device), dim=1)
        # find P(C|U,X) for each x in observation
        full_p_c_ux = torch.permute(p_c_ux, (2, 0, 1))[x_obs]
        # compute Q(Y|X)
        q_y_x = torch.einsum('yuc,dcu,du->dy', p_y_uc, full_p_c_ux, q_u_x)
        # normalize Q(Y|X) because equation is proportional to
        q_y_x = q_y_x / torch.sum(q_y_x, dim=1).unsqueeze(1)

    else:
        # compute Q(Y|X)
        q_y_x = torch.einsum('yuc,dcu,du->dy', p_y_uc, p_c_ux, q_u_x)
        # normalize Q(Y|X) because equation is proportional to
        q_y_x = q_y_x / torch.sum(q_y_x, dim=1).unsqueeze(1)

    return q_y_x


def RMSE(vec1, vec2):
    '''
    compute root mean square error between two 1D vectors
    '''
    loss = nn.MSELoss()
    return torch.sqrt(loss(vec1, vec2)).cpu().numpy()


def KL_div(P, Q):
    '''
    KL divergence between P and Q where both are discrete. Both are joint probabilities
    '''
    Q[Q == 0.0] = 1e-20
    P[P == 0.0] = 1e-20
    return np.sum(P * np.log(P)) - np.sum(P * np.log(Q))


def conditional_KL(P_cond, Q_cond, P_joint):
    '''
    KL div between two codnitional distributions P_cond, Q_cond
    '''
    Q_cond[Q_cond == 0.0] = 1e-20
    P_cond[P_cond == 0.0] = 1e-20
    return np.sum(P_joint * np.log(P_cond)) - np.sum(P_joint * np.log(Q_cond))


def read_dataframe_into_dict(folder_id, filename_source, filename_target, data_type, dset):
    '''
    read from folder_id and into dataframe. then convert to dict
    '''
    data_df_source = pd.read_csv(os.path.join(folder_id, filename_source))
    data_df_target = pd.read_csv(os.path.join(folder_id, filename_target))
    data_dict_source = extract_from_df_nested(data_df_source)
    data_dict_target = extract_from_df_nested(data_df_target)
    data_dict_all = dict(source=data_dict_source, target=data_dict_target) # ['train', 'val', 'test'] ['u', 'x', 'w', 'c', 'c_logits', 'y', 'y_logits', 'y_one_hot', 'w_binary', 'w_one_hot', 'u_one_hot']
    print('data dict all',data_dict_all['target']['train'].keys())
    print('x shape', data_dict_all['source']['train']['x'].shape)

    # put dataframe data into dictionary so you can convert easily to
    target_obs = {}
    num_obs = len(data_dict_all['target']['train']['x'])
    target_obs['X'] = torch.tensor(data_dict_all['target'][dset]['x'], dtype=data_type)
    target_obs['U'] = torch.tensor(data_dict_all['target'][dset]['u'], dtype=int)
    target_obs['C'] = torch.tensor(data_dict_all['target'][dset]['c'], dtype=int)
    target_obs['Y'] = torch.tensor(data_dict_all['target'][dset]['y_one_hot'], dtype=int)
    target_obs['W'] = torch.tensor(data_dict_all['target'][dset]['w_one_hot'], dtype=data_type)
    target_obs['num_obs'] = num_obs

    source_obs = {}
    num_obs = len(data_dict_all['source']['train']['x'])
    source_obs['X'] = torch.tensor(data_dict_all['source'][dset]['x'], dtype=data_type)
    source_obs['W'] = torch.tensor(data_dict_all['source'][dset]['w_one_hot'], dtype=data_type)
    source_obs['Y'] = torch.tensor(data_dict_all['source'][dset]['y_one_hot'], dtype=int)
    source_obs['C'] = torch.tensor(data_dict_all['source'][dset]['c'], dtype=int)
    source_obs['U'] = torch.tensor(data_dict_all['source'][dset]['u'], dtype=int)
    source_obs['num_obs'] = num_obs

    return source_obs, target_obs, num_obs, data_dict_all



def create_entropy_tmp_schd(paramDict, device):
    '''
    entropy tmp is a factor in front of the variational entropy term of the free energy (should decay to 1.0)
    '''
    if 'entropy_tmp_schedule' in paramDict.keys():
        percent_end, start = paramDict['entropy_tmp_schedule']
        entropy_tmp_schedule = torch.ones(paramDict['epochs']).to(device)
        end_epoch = math.floor(paramDict['epochs'] * percent_end)
        schd = torch.linspace(start=start, end=1.0, steps=end_epoch)
        entropy_tmp_schedule[0:end_epoch] = schd
    else:
        entropy_tmp_schedule = None

    return entropy_tmp_schedule

def create_prior_tmp_schd(paramDict, device):
    '''
    prior tmp is a factor in front of the KL between variational entropy and prior (should decay to 1.0)
    '''
    if 'prior_tmp_schedule' in paramDict.keys():
        percent_end, start = paramDict['prior_tmp_schedule']
        prior_tmp_schedule = torch.ones(paramDict['epochs']).to(device)
        end_epoch = math.floor(paramDict['epochs'] * percent_end)
        schd = torch.linspace(start=start, end=1.0, steps=end_epoch)
        prior_tmp_schedule[0:end_epoch] = schd
    else:
        prior_tmp_schedule = None

    return prior_tmp_schedule


#########################################################################################
# Evaluation Metrics
#########################################################################################


# Define sklearn evaluation metrics
def soft_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    return sklearn.metrics.accuracy_score(y_true, y_pred >= threshold, **kwargs)


def soft_balanced_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred >= threshold, **kwargs)


def log_loss64(y_true, y_pred, **kwargs):
    return sklearn.metrics.log_loss(y_true, y_pred.astype(np.float64), **kwargs)


def evaluate_clf(evals_sklearn, y_pred_target, data_dict_all, y_pred_source=None):
    result_dict = {}
    for metric in evals_sklearn.keys():
        result_dict[metric] = {}

    y_pred_target = y_pred_target.cpu().numpy()[:, 1]
    y_true_target = data_dict_all['target']['test']['y']
    if y_pred_source is not None:
        y_pred_source = y_pred_source.cpu().numpy()[:, 1]
        y_true_source = data_dict_all['source']['test']['y']

    print('y true target example', y_true_target[:10])

    for metric in evals_sklearn.keys():
        if y_pred_source is not None:
            result_dict[metric]['eval_on_source'] = evals_sklearn[metric](y_true_source, y_pred_source)
        result_dict[metric]['eval_on_target'] = evals_sklearn[metric](y_true_target, y_pred_target)
    return result_dict


#########################################################################################
# Evaluation Metrics
#########################################################################################


# Convert data to dataframe format
def pack_to_df(samples_dict):
    return pd.concat({key: get_squeezed_df(value) for key, value in samples_dict.items()}).reset_index(level=-1,
                                                                                                       drop=True).rename_axis(
        'partition').reset_index()


# Extract dataframe format back to dict format
def extract_from_df(samples_df, cols=['u', 'x', 'w', 'c', 'c_logits', 'y', 'y_logits', 'y_one_hot', 'w_binary', 'w_one_hot',
                          'u_one_hot', 'x_scaled']):
    """
    Extracts dict of numpy arrays from dataframe
    """
    result = {}
    for col in cols:
        if col in samples_df.columns:
            result[col] = samples_df[col].values
        else:
            match_str = f"^{col}_\d+$"
            r = re.compile(match_str, re.IGNORECASE)
            matching_columns = list(filter(r.match, samples_df.columns))
            if len(matching_columns) == 0:
                continue
            result[col] = samples_df[matching_columns].to_numpy()
    return result


def extract_from_df_nested(samples_df, cols=['u', 'x', 'w', 'c', 'c_logits', 'y', 'y_logits', 'y_one_hot', 'w_binary', 'w_one_hot',
                                 'u_one_hot', 'x_scaled']):
    """
    Extracts nested dict of numpy arrays from dataframe with structure {domain: {partition: data}}
    """
    result = {}
    if 'domain' in samples_df.keys():
        for domain in samples_df['domain'].unique():
            result[domain] = {}
            domain_df = samples_df.query('domain == @domain')
            for partition in domain_df['partition'].unique():
                partition_df = domain_df.query('partition == @partition')
                result[domain][partition] = extract_from_df(partition_df, cols=cols)
    else:
        for partition in samples_df['partition'].unique():
            partition_df = samples_df.query('partition == @partition')
            result[partition] = extract_from_df(partition_df, cols=cols)
    return result


def vector_x_dataset(source_obs, target_obs, batch_size, shuffle=False):
    '''
    make a dataset with shuffle on
    '''
    source_dset = DiscreteMNISTDataset(source_obs['U'], source_obs['Y'], c=source_obs['C'],
                                       x_images=source_obs['X'], w_images=source_obs['W'])
    source_dloader = DataLoader(source_dset, batch_size, num_workers=1, pin_memory=True, shuffle=shuffle)
    target_dset = DiscreteMNISTDataset(target_obs['U'], target_obs['Y'], c=target_obs['C'],
                                       x_images=target_obs['X'], w_images=target_obs['W'])
    target_dloader = DataLoader(target_dset, batch_size, num_workers=1, pin_memory=True, shuffle=False)

    return source_dloader, target_dloader

