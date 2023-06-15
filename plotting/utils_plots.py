'''
plotting functions
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os




def plot_loss(plot_folder, name, loss_tot, title=None):
    plt.figure()
    plt.plot(loss_tot)
    if title is not None:
        plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Free Energy')
    plt.savefig(plot_folder + name + '.png')
    plt.close()


def plot_average_posterior_to_true(plot_folder, q_varitional_probs, true_latent, latent_dim, permute):
    '''
    plot permute latent average posterior \sum p(U|Y,X,W,C) for each true latent
    '''
    # subplot_titles = np.array(map(str, np.arange(num_odors)))
    subplot_titles = np.char.mod('latent %d', np.arange(latent_dim))
    # print('subplot titles',subplot_titles, latent_marginals.shape)
    fig = make_subplots(rows=1, cols=latent_dim, subplot_titles=subplot_titles)

    for cur_latent in range(latent_dim):
        # Prediction digit = digit_cur
        idx = (true_latent == cur_latent)

        # How many times 'digit_cur' from prediction is mapped to each digit in label.
        # scor_tot[cur_latent, :] = (labels[idx].unsqueeze(dim=0) == cur_latent.unsqueeze(dim=1)).sum(1)
        posterior = (q_varitional_probs[idx].sum(0)/idx.sum())[permute]

        fig.add_trace(go.Bar(x=np.arange(latent_dim), y=posterior), row=1, col=(cur_latent + 1))
    # fig.update_layout(
    #     # title_text='Sampled Results',  # title of plot
    #     xaxis_title_text='track position',  # xaxis label
    #     yaxis_title_text='activity of latent'  # yaxis label
    #     # bargap=0.2  # gap between bars of adjacent location coordinates
    # )
    fig.write_image(plot_folder + "average_latent_posterior_vs_true.png")

def plot_KL(plot_folder, name, KL_list, sample_KL_list, size_inds, data_size, diff_colors=False):
    '''
    plot KL divergence with dataset size
    '''
    # plot error histogram
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Plotly + px.colors.qualitative.Plotly
    if diff_colors:
        fig.add_trace(go.Scatter(x=size_inds,y=KL_list, name='learned', mode='markers',
                             marker=dict(size=10, color=colors, line=dict(width=1))))
    else:
        fig.add_trace(go.Scatter(x=size_inds,y=KL_list, name='learned', mode='markers',
                             marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))))
    if len(sample_KL_list) > 0:
        fig.add_trace(go.Scatter(x=size_inds, y=sample_KL_list, name='sampled'))
    fig.update_layout(
        title_text='KL divergence',  # title of plot
        xaxis_title_text='dataset size',  # xaxis label
        yaxis_title_text='KL',  # yaxis label
        xaxis=dict(tickmode='array',tickvals = np.arange(len(KL_list)), ticktext=data_size),
        showlegend=True
    )
    fig.write_image(plot_folder + name+".png")

def plot_RMSE(plot_folder, name, RMSE_learn, RMSE_sample, RMSE_P, RMSE_NN_pred, size_inds, data_size, diff_colors=False):
    '''
    plot KL divergence with dataset size
    '''
    # plot error histogram
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Plotly + px.colors.qualitative.Plotly
    if diff_colors:
        fig.add_trace(go.Scatter(x=size_inds,y=RMSE_learn, name='learned', mode='markers',
                             marker=dict(size=10, color=colors, line=dict(width=1))))
    else:
        fig.add_trace(go.Scatter(x=size_inds,y=RMSE_learn, name='learned', mode='markers',
                             marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))))
    fig.add_trace(go.Scatter(x=size_inds, y=RMSE_sample, name='sampled'))
    # if len(RMSE_P) > 0:
    #     fig.add_trace(go.Scatter(x=size_inds, y=RMSE_P, name='P(Y|X)'))
    if len(RMSE_NN_pred) > 0:
        fig.add_trace(go.Scatter(x=size_inds, y=RMSE_NN_pred, name='NN P(Y|X)', mode='markers',
                                 marker=dict(size=10, line=dict(width=1))))
    fig.update_layout(
        title_text='RMSE',  # title of plot
        xaxis_title_text='dataset size',  # xaxis label
        yaxis_title_text='RMSE',  # yaxis label
        xaxis=dict(tickmode='array',ticktext=data_size)
    )
    fig.write_image(plot_folder + name+".png")


def plot_digits(plot_folder, name, observations, obs_label):
    cols = 6
    rows = 4
    fig, axs = plt.subplots(rows, cols, figsize=(12, 9))

    # for obs in observations:
    for x in range(rows):
        for y in range(cols):
            #rnd_idx = range(len(trainset.data))
            idx = int(x * rows + y)
            axs[x, y].set_title(obs_label[idx])
            axs[x, y].imshow(observations[idx][0])
            axs[x, y].set_axis_off()

    plt.savefig(plot_folder + name + '.png')
        # break


def plot_KL_vs_free_energy(plot_folder, name, KL_list, free_energy, size_inds, data_size):
    '''
    plot KL divergence with dataset size
    '''
    # plot error histogram
    fig = go.Figure()
    for dsize in np.unique(size_inds):
        mask = size_inds == dsize
        KL_mask = np.array(KL_list)[mask]
        FE_mask = np.array(free_energy)[mask]
        FE_mask = FE_mask - np.min(FE_mask)
        sort_ind = np.argsort(FE_mask)
        fig.add_trace(go.Scatter(x=FE_mask[sort_ind],y=KL_mask[sort_ind], name=str(data_size[dsize]),
                                 marker=dict(size=10, line=dict(width=1))))
    fig.update_layout(
        title_text='KL divergence',  # title of plot
        xaxis_title_text='free energy',  # xaxis label
        yaxis_title_text='KL',  # yaxis label
        #xaxis=dict(tickmode='array',tickvals = np.arange(len(KL_list)), ticktext=data_size),
        showlegend=True
    )
    fig.write_image(plot_folder + name+".png")

def plot_KL_vs_FE_no_min(plot_folder, name, KL_list, free_energy, size_inds, data_size):
    '''
    plot KL divergence with dataset size
    '''
    # plot error histogram
    fig = go.Figure()
    for dsize in np.unique(size_inds):
        mask = size_inds == dsize
        KL_mask = np.array(KL_list)[mask]
        FE_mask = np.array(free_energy)[mask]
        # FE_mask = FE_mask - np.min(FE_mask)
        sort_ind = np.argsort(FE_mask)
        fig.add_trace(go.Scatter(x=FE_mask[sort_ind],y=KL_mask[sort_ind], name=str(data_size[dsize]),
                                 marker=dict(size=10, line=dict(width=1))))
    fig.update_layout(
        title_text='KL divergence',  # title of plot
        xaxis_title_text='free energy',  # xaxis label
        yaxis_title_text='KL',  # yaxis label
        #xaxis=dict(tickmode='array',tickvals = np.arange(len(KL_list)), ticktext=data_size),
        showlegend=True
    )
    fig.write_image(plot_folder + name+".png")



def plot_var_probs(plot_folder, name, q_varitional_probs, true_obs, obs_dim):
    '''
    plot variational posterior probs for each value of true_obs
    '''
    # subplot_titles = np.array(map(str, np.arange(num_odors)))
    subplot_titles = np.char.mod('%d', np.arange(obs_dim))
    # print('subplot titles',subplot_titles, latent_marginals.shape)
    fig = make_subplots(rows=1, cols=obs_dim, subplot_titles=subplot_titles)

    for cur_x in range(obs_dim):
        # Prediction digit = digit_cur
        idx = (true_obs == cur_x)

        # How many times 'digit_cur' from prediction is mapped to each digit in label.
        # scor_tot[cur_latent, :] = (labels[idx].unsqueeze(dim=0) == cur_latent.unsqueeze(dim=1)).sum(1)
        for ind in range(20):
            posterior = q_varitional_probs[idx]

            # fig.add_trace(go.Bar(x=np.arange(latent_dim), y=posterior), row=1, col=(cur_latent + 1))
            fig.add_trace(go.Scatter(x=np.arange(q_varitional_probs.shape[-1]), y=posterior[ind]), row=1, col=(cur_x + 1))
        fig.update_yaxes(range = [0.0,1.0])
    fig.update_layout(
        showlegend=False,

    )
    #     # title_text='Sampled Results',  # title of plot
    #     xaxis_title_text='track position',  # xaxis label
    #     yaxis_title_text='activity of latent'  # yaxis label
    #     # bargap=0.2  # gap between bars of adjacent location coordinates
    # )
    fig.write_image(plot_folder + name + ".png")

def plot_factor_probs_together(plot_folder, name, q_varitional_probs, true_obs_1, obs_dim_1, true_obs_2, obs_dim_2):
    '''
    plot factors for each value of true_obs_1 and 2
    '''
    # subplot_titles = np.array(map(str, np.arange(num_odors)))
    # subplot_titles = np.char.mod('%d', np.arange(obs_dim))
    # print('subplot titles',subplot_titles, latent_marginals.shape)
    fig = make_subplots(rows=obs_dim_2, cols=obs_dim_1)#, subplot_titles=subplot_titles)

    for cur_1 in range(obs_dim_1):
        for cur_2 in range(obs_dim_2):
            idx = (true_obs_1 == cur_1) & (true_obs_2 == cur_2)
            posterior = q_varitional_probs[idx]
            num_plots = len(posterior)
            if num_plots > 20:
                num_plots = 20
            for ind in range(num_plots):
                fig.add_trace(go.Scatter(x=np.arange(q_varitional_probs.shape[-1]), y=posterior[ind]), row=(cur_2 + 1), col=(cur_1 + 1))
            fig.update_yaxes(range = [0.0,1.0])
    fig.update_layout(
        showlegend=False,

    )
    #     # title_text='Sampled Results',  # title of plot
    #     xaxis_title_text='track position',  # xaxis label
    #     yaxis_title_text='activity of latent'  # yaxis label
    #     # bargap=0.2  # gap between bars of adjacent location coordinates
    # )
    fig.write_image(plot_folder + name + ".png")


def plot_prior_probs(plot_folder, name, prior):
    '''
    plot prior probs
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(prior.shape[-1]), y=prior))
    fig.update_yaxes(range = [0.0,1.0])
    fig.update_layout(
        showlegend=False,
    )
    fig.write_image(plot_folder + name + ".png")


def plot_factor_examples(plot_folder, name, factor_x, factor_w, obs_loader, true_obs_c, true_obs_y):
    '''
    plot a bunch of digit obs and the factors
    '''
    if os.path.exists(plot_folder + name):
        print('overwrite')
    else:
        os.makedirs(plot_folder + name)
    for batch_idx, observations in enumerate(obs_loader):
        for idx in range(len(observations['X'])):
            if idx > 20:
                break
            fig = make_subplots(rows=2, cols=2)#, subplot_titles=subplot_titles)
            fig.add_trace(px.imshow(observations['X'][idx,0].numpy()).data[0], row=1, col=1)
            fig.update_yaxes(autorange='reversed')
            fig.add_trace(px.imshow(observations['W'][idx,0].numpy()).data[0], row=1, col=2)
            fig.update_yaxes(autorange='reversed')
            fig.add_trace(go.Scatter(x=np.arange(factor_x[idx].shape[-1]), y=factor_x[idx]), row=2, col=1)
            fig.add_trace(go.Scatter(x=np.arange(factor_w[idx].shape[-1]), y=factor_w[idx]), row=2, col=2)
            fig.update_layout(
                showlegend=False,
                title_text='C:' + str(true_obs_c[idx].item()) + ' Y:' + str(true_obs_y[idx].item()),  # title of plot
            )

            fig.write_image(plot_folder + name + '/example_' + str(idx) + ".png")

        break


def plot_metric(plot_folder, name, LAT_df, x_axis):
    '''
    plot KL divergence with dataset size
    '''
    # all_methods = ['RPM','erm_source','erm_target','vae_graph','vae_vanilla']
    all_methods = np.unique(LAT_df['method']) #['RPM','erm_source','erm_target','vae_graph','vae_vanilla']
    # all_methods = [all_methods[1],all_methods[0]]

    # plot error histogram
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Plotly + px.colors.qualitative.Plotly
    for eval_set in np.unique(LAT_df['eval_set']):
        for metric in ['cross-entropy','accuracy','balanced_accuracy','auc']:
            fig = go.Figure()
            # for indm,method in enumerate(np.unique(LAT_df['method'])):
            for indm,method in enumerate(all_methods):
                next_color = colors[indm]
                subset = LAT_df.loc[(LAT_df['method'] == method) & (LAT_df['eval_set'] == eval_set)]
                z = subset['x_dim']
                x_unique, x_inv = np.unique(subset['x_dim'], return_inverse=True)
                y = subset[metric]
                fig.add_trace(go.Scatter(x=x_inv,y=y, mode='markers', line=dict(color=next_color),
                                 marker=dict(size=5), showlegend=False))
                means = []
                stds = []
                x_ticks = []
                # x_ticks = ['2E3',]
                for x_dim in x_unique:
                    means = means + [np.mean(y[z==x_dim])]
                    stds = stds + [np.std(y[z==x_dim])]
                    x_ticks = x_ticks + ["{:.0E}".format(x_dim)] #[str(int(x_dim))]
                means = np.array(means)
                stds = np.array(stds)
                xnum=np.arange(len(x_unique))
                # if method == 'ERM-SOURCE':
                #     method = 'erm_source'
                fig.add_trace(go.Scatter(x=xnum, y=means, mode='lines', line=dict(color=next_color), name=method))
                fig.add_trace(go.Scatter(x=xnum, y=(means + stds), mode='lines', line=dict(color=next_color,width=0.1), showlegend=False))
                fig.add_trace(go.Scatter(x=xnum, y=(means - stds), fill='tonexty', mode='lines', line_color=next_color, line=dict(color=next_color,width=0.1), showlegend=False))
            fig.update_layout(
                title_text=metric + ' ' + eval_set,  # title of plot
                xaxis_title_text=x_axis,  # xaxis label
                yaxis_title_text=metric,  # yaxis label
                # xaxis=dict(tickmode='array', ticktext=data_size)
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = xnum, #[1, 3, 5, 7, 9, 11],
                    ticktext = x_ticks # np.char.mod('%d', x_unique)#['One', 'Three', 'Five', 'Seven', 'Nine', 'Eleven']
                )
            )
            fig.write_image(plot_folder + name + '_' + metric + '_' + eval_set + ".png")

    # fig.add_trace(go.Scatter(x=size_inds, y=RMSE_sample, name='sampled'))
    # # if len(RMSE_P) > 0:
    # #     fig.add_trace(go.Scatter(x=size_inds, y=RMSE_P, name='P(Y|X)'))
    # if len(RMSE_NN_pred) > 0:
    #     fig.add_trace(go.Scatter(x=size_inds, y=RMSE_NN_pred, name='NN P(Y|X)', mode='markers',
    #                              marker=dict(size=10, line=dict(width=1))))
