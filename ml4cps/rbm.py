"""
    The module implements the novel DENTA algorithm for the learning of hybrid automata from data.

    Author:
    Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
"""

import pprint
import time as tm
from torch.utils.data import DataLoader
from datetime import timedelta
import mlflow
import torch
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
from ml4cps import tools, examples, vis
from plotly import graph_objects as go
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class XBinaryRBM(ABC, nn.Module):
    """
    It is an "abstract" class for modeling RBM with binary latent units.
    """
    def __init__(self, n_visible, n_hidden, device='cpu'):
        """
        Initialize the X-Binary RBM.
        :param n_visible: Number of visible units
        :param n_hidden: Number of hidden units
        """
        super(XBinaryRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Initialize weights and biases
        self.weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01).to(device)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible)).to(device)
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden)).to(device)

        self.learning_curve = []
        self.valid_curve = []
        self.num_epoch = 0

    def _get_progress_rbm(self, d):
        with torch.no_grad():
            r, h = self.recon(d)
            e = self.free_energy(d)
            progress = dict(MSE=mse(d.view(d.shape[0], -1), r).item(),
                            Sparsity=sparsity(h).item(),
                            Energy=torch.mean(e).item(),
                            Weights=torch.mean(torch.abs(self.weights)).item(),
                            VisBias=torch.mean(torch.abs(self.visible_bias)).item(),
                            HidBias=torch.mean(torch.abs(self.hidden_bias)).item())
        return progress

    def v2h(self, visible, pre_sigmoid=False):
        visible = visible.view(visible.size(0), -1)
        if pre_sigmoid:
            return torch.matmul(visible, self.weights) + self.hidden_bias
        else:
            return torch.sigmoid(torch.matmul(visible, self.weights) + self.hidden_bias)

    def sample_h(self, hidden_probs):
        """
        Sample hidden states given visible states.
        :param visible: Input visible states
        :return: Binary hidden states and probabilities
        """
        hidden_states = torch.bernoulli(hidden_probs)
        return hidden_states

    @abstractmethod
    def sample_x(self, hidden):
        pass

    def recon_error(self, data, input=None, per_point=False, round=False):
        if input is None:
            input = data
        recon, _ = self.recon(input, round=round)
        if per_point:
            dim = tuple(range(1, data.dim()))
        else:
            dim = None
        squared_error = torch.mean(torch.square(data - recon), dim=dim)
        return squared_error

    @abstractmethod
    def free_energy(self, x):
        """
        Compute the free energy of the visible units.
        :param x: Visible states.
        :return: Vector of free energy scalars.
        """
        pass

    def forward(self, visible):
        """
        Forward pass: Gibbs sampling for reconstruction.
        :param visible: Input visible states
        :return: Reconstructed visible states
        """
        hidden_prob = self.v2h(visible)
        reconstructed_visible = self.h2v(hidden_prob)
        return reconstructed_visible

    def recon(self, v, round=False):
        h = self.v2h(v)
        v = self.h2v(h)
        if round:
            return v, torch.round(h)
        else:
            return v, h

    def generate(self, num_examples, num_steps=10):
        with torch.no_grad():
            if isinstance(self, FractionalBinaryRBM):
                x = self.min_values + self.max_values * torch.rand(num_examples, int(self.n_visible/self.n_fractions))
                x = self.prepare_input(x)
            elif isinstance(self, ThreeWayBinaryBinaryRBM):
                # Initialization
                # For v (visible units): Start with random binary values or use data input
                v_init = torch.randint(0, 2, (num_examples, self.n_visible)).float()

                # For u (third-layer units): Random binary values (e.g., Bernoulli distribution)
                u_init = torch.randint(0, 2, (num_examples, self.n_auxiliary)).float()
            else:
                x = torch.randn([num_examples, self.n_visible])


            for k in range(num_steps):
                ph = self.v2h(x)
                h = self.sample_h(ph)
                px = self.h2v(h)
                x = self.sample_x(px)
        return x

    def prepare_input(self, d):
        return d

    def decode_input(self, x):
        return x

    def train_rbm(self, train_data, valid_data, learning_rule='cd', min_epoch=0, max_epoch=10, weight_decay=0., lr=0.01,
                  batch_size=128, shuffle=True, num_gibbs=1, verbose=True, early_stopping=False, early_stopping_patience=3,
                  use_probability_last_x_update=True, log_mlflow=False, denoising=False, sparsity_weight=0,
                  sparsity_target=0.1, optimizer="RMSprop"):
        """
        param: train_data: Training data, list of tensors.
        param: valid_data: Validation data, list of tensors.
        """
        if log_mlflow:
            mlflow.log_param("min_epoch", min_epoch)
            mlflow.log_param("max_epoch", max_epoch)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("shuffle", shuffle)
            mlflow.log_param("weight_decay", weight_decay)
            mlflow.log_param("early_stopping_patience", early_stopping_patience)
            mlflow.log_param("early_stopping", early_stopping)
            mlflow.log_param("sparsity_weight", sparsity_weight)
            mlflow.log_param("sparsity_target", sparsity_target)
            mlflow.log_param("lr", lr)
            mlflow.log_param("num_gibbs", num_gibbs)
            mlflow.log_param("optimizer", optimizer)

        # train_data = [self.prepare_input(td) for td in train_data]
        # valid_data = [self.prepare_input(vd) for vd in valid_data]
        train_data = torch.vstack(train_data).float()
        valid_data = torch.vstack(valid_data).float()

        # if valid_data is not None:
        #     self.valid_curve.append(self._get_progress_rbm(valid_data, level=level))
        #
        # self.learning_curve.append(self._get_progress_rbm(train_data, level=level))

        data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
        if optimizer == "SGD":
            opt = torch.optim.SGD(self.parameters(), weight_decay=weight_decay, lr=lr)
        elif optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.parameters(), weight_decay=weight_decay, lr=lr)
        else:
            raise NotImplementedError()

        t_start = tm.time()
        for epoch in range(1, max_epoch + 1):
            if verbose:
                print(f'Epoch {epoch} started...')
            for i, d in enumerate(data_loader):
                xk = d
                if denoising:
                    pass
                    # xk = torch.normal(mean=xk, std=torch.exp(self.log_sigma_x))
                x0 = d
                if learning_rule == 'cd':
                    opt.zero_grad()
                    cd_loss = self.contrastive_divergence(x0, num_gibbs=num_gibbs,
                                                          use_probability_last_x_update=use_probability_last_x_update)
                    cd_loss.backward()
                    opt.step()
                elif learning_rule == 'sm':
                    r = self.recon(xk)
                    loss = (r - x0).pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                elif learning_rule == 'dsm':
                    x0noise = torch.normal(torch.zeros_like(x0))
                    loss = self.dsm_loss(x0, x0noise)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                if verbose and verbose > 1:
                    with torch.no_grad():
                        progress = self._get_progress_rbm(train_data)
                        print(f'\n############### BATCH {i} ###############')
                        print('Train: ')
                        pprint.pp(progress)

            with torch.no_grad():
                progress = self._get_progress_rbm(train_data)
                if verbose:
                    print(f'\n############### Epoch {epoch} ###############')
                    print('Train: ')
                    pprint.pp(progress)
                self.learning_curve.append(progress)
            if valid_data is not None:
                with torch.no_grad():
                    progress = self._get_progress_rbm(valid_data)
                    self.valid_curve.append(progress)
                if verbose:
                    print('Valid: ')
                    pprint.pp(progress)

                if early_stopping and epoch > min_epoch and epoch > early_stopping_patience:
                    valid_metrics = np.array([v['MSE'] for v in self.valid_curve[-early_stopping_patience - 1:]])
                    if np.all(valid_metrics[1:] > valid_metrics[0]):
                        print('Early stop after valid metrics: ', valid_metrics)
                        break

        self.eval()
        self.num_epoch = epoch
        print('Training finished after ', timedelta(seconds=tm.time() - t_start))

    def contrastive_divergence(self, d, num_gibbs=1, use_probability_last_x_update=False, auxiliary=None):
        """
        Perform Contrastive Divergence (CD-k) to train the RBM.
        :param visible: Input visible states
        :param num_gibbs: Number of Gibbs sampling steps
        :return: CD loss
        """

        with torch.no_grad():
            # Positive phase
            visible = d
            hidden_states = self.sample_h(self.v2h(visible))

            # Gibbs sampling
            for _ in range(num_gibbs):
                visible_states = self.sample_x(self.h2v(hidden_states))
                hidden_probs = self.v2h(visible_states)
                hidden_states = self.sample_h(hidden_probs)

            # Negative phase
            if use_probability_last_x_update:
                negative_visible_states = self.h2v(hidden_states)
            else:
                negative_visible_states = self.sample_x(self.h2v(hidden_states))

        loss = torch.mean(self.free_energy(visible) - self.free_energy(negative_visible_states))
        return loss

    def plot_learning_curve(self):
        return vis.plot_timeseries([pd.DataFrame(self.learning_curve), pd.DataFrame(self.valid_curve)],
                             title='Learning curve', names=['Train', 'Valid'], xaxis_title='Epoch')

    def fit(self, data, epochs=10, batch_size=64, k=1, learning_rate=0.01):
        """
        Train the RBM using Contrastive Divergence.
        :param data: Input data (torch.Tensor)
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param k: Number of Gibbs sampling steps
        :param learning_rate: Learning rate
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                optimizer.zero_grad()
                loss = self.contrastive_divergence(batch, k)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def plot_input_space(self, data=None, samples=None, show_gaussian_components=False, data_limit=10000,
                         xmin=None, xmax=None, ymin=None, ymax=None, figure_width=600, figure_height=600,
                         show_axis_titles=True, show_energy_contours=False, showlegend=True,
                         show_recon_error_contours=False, ncontours=None, plot_code_positions=True,
                         show_recon_error_heatmap=False, plot_bias_vector=False, show_reconstructions=False,
                         plot_separation_lines=False,
                         samples_opacity=0.2, **kwargs):
        fig = go.Figure()

        if plot_separation_lines:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
            with torch.no_grad():
                binary_codes = torch.round(self.v2h(torch.Tensor(d))).numpy()

                # Step 1: Find unique binary combinations
                unique_combinations, indices = np.unique(binary_codes, axis=0, return_inverse=True)

                # Step 2: Generate colors for each unique combination
                num_combinations = len(unique_combinations)
                colors = plt.cm.get_cmap('viridis', num_combinations)(range(num_combinations))

                # Map each row in the binary_codes array to its corresponding color
                color_map = [colors[idx] for idx in indices]

                # Convert RGBA colors to Plotly-compatible hex
                color_map_hex = [f"rgba({r * 255:.0f},{g * 255:.0f},{b * 255:.0f},{a:.2f})" for r, g, b, a in color_map]

                # Step 3: Create scatter plot
                scatter = go.Scatter(
                        x=d[:, 0],
                        y=d[:, 1],
                        mode='markers',
                        opacity=0.5,
                        marker=dict(
                            symbol='square',
                            size=3,
                            color=color_map_hex  # Assign colors dynamically
                        ),
                    )
                fig.add_trace(scatter)

        if show_recon_error_heatmap:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
            with torch.no_grad():
                fe = self.recon_error(torch.Tensor(d)).numpy()

            trace = go.Heatmap(x=x, y=y, z=np.reshape(fe, xv.shape),
                               name="Reconstruction Error", showlegend=True, showscale=False)
            fig.add_trace(trace)

        if show_recon_error_contours and data.shape[1] == 2:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
            with torch.no_grad():
                fe = self.recon_error(torch.Tensor(d), per_point=True).numpy()
                fe = np.reshape(fe, xv.shape)

                trace = go.Contour(x=x, y=y, z=fe, contours=dict(coloring='lines'), name="Reconstruction Error",
                                   showlegend=True, showscale=False, ncontours=ncontours)
                fig.add_trace(trace)

        if show_energy_contours:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])

            d = self.prepare_input(torch.Tensor(d))
            fe = self.free_energy(d).detach().numpy()

            trace = go.Contour(x=x, y=y, z=np.reshape(fe, xv.shape), contours=dict(coloring='lines'),
                               name="Free energy", ncontours=ncontours, showlegend=True, showscale=False)
            fig.add_trace(trace)

        if data is not None:
            if data_limit is not None and data.shape[0] > data_limit:
                data = data.sample(data_limit)
            fig.add_trace(vis.plot2d(data, data.columns[0], data.columns[1], name='Data',
                                     marker=dict(size=3, opacity=0.2, color='MediumPurple')))
            if show_reconstructions:
                recon, h_recon = self.recon(torch.tensor(data.values).float(), round=True)
                recon = pd.DataFrame(recon.detach().numpy())
                fig.add_trace(vis.plot2d(recon, recon.columns[0], recon.columns[1], name='Reconstruction',
                                         marker=dict(size=3, opacity=0.2, color='limegreen')))
        if samples is not None:
            col1 = samples.columns[0]
            col2 = samples.columns[1]
            samples = self.decode_input(torch.tensor(samples.to_numpy())).detach().numpy()
            samples = pd.DataFrame(samples, columns=[col1, col2])
            fig.add_trace(vis.plot2d(samples, col1, col2, name='Samples',
                                     marker=dict(size=3, opacity=samples_opacity, color='darkgreen')))

        if show_axis_titles:
            fig.update_layout(
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
            )
        if plot_code_positions:
            num_h = self.n_hidden

            if isinstance(self, FractionalBinaryRBM):
                num_v = int(self.n_visible / self.n_fractions)
            else:
                num_v = self.n_visible
            num_components = 2 ** num_h
            # Initialize
            means = np.zeros((num_components, num_v))
            hid_states = np.zeros((num_components, num_h))
            for i in range(0, num_components):
                hs = list(bin(i)[2:])
                hid_states[i, -len(hs):] = hs
                hs = hid_states[[i], :]
                # Calc means
                with torch.no_grad():
                    mean = self.h2v(torch.Tensor(hs))
                    mean = self.decode_input(mean)
                    means[i, :] = mean.detach().numpy().reshape(1, -1)

            hm_mapping = dict()
            for h, m in zip(list(hid_states), list(means)):
                hm_mapping[str(h)] = m
            for i in range(means.shape[0]):
                mean = means[i, :]
                hid = hid_states[i, :]
                for i, hi in enumerate(hid):
                    if hi == 1:
                        hid_prev = hid.copy()
                        hid_prev[i] = 0
                        mean_start = hm_mapping[str(hid_prev)]
                        fig.add_annotation(xref="x", yref="y", axref="x", ayref="y", ax=mean_start[0], ay=mean_start[1],
                                           x=mean[0], y=mean[1], showarrow=True, arrowhead=2, arrowsize=1.5)

            fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers', name='Codes',
                                     textfont_size=12, textposition="top left", marker_color='orange', marker_size=4))
        if plot_bias_vector:
            bx = self.h2v(torch.tensor(np.zeros((1, self.n_hidden)), requires_grad=False).float()) # self.visible_bias.detach()[None] #* np.exp(self.log_sigma_x.detach().numpy()).flatten()
            bx = self.decode_input(bx.detach()).numpy().flatten()
            fig.add_annotation(xref="x", yref="y", axref="x", ayref="y", x=bx[0], y=bx[1], ax=0, ay=0,
                               showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1, arrowcolor="#636363")
        if show_gaussian_components and isinstance(self, GaussianBinaryRBM):
            weights, means, gmm_sigmas, hid_states, Z = self.gmm_model()
            hm_mapping = dict()
            for h, m in zip(list(hid_states), list(means)):
                hm_mapping[str(h)] = m
            for i in range(weights.shape[0]):
                weight = weights[i, 0]
                mean = means[i, :]
                sigma = gmm_sigmas[i, :]
                hid = hid_states[i, :]
                fig.add_shape(type="circle",
                              xref="x", yref="y",
                              x0=mean[0] - 2 * sigma[0], y0=mean[1] - 2 * sigma[1],
                              x1=mean[0] + 2 * sigma[0], y1=mean[1] + 2 * sigma[1],
                              # opacity=weight/max(max(weights)),
                              fillcolor='rgba(23, 156, 125, {:.2f})'.format(0.7 * weight / max(max(weights))),
                              line_color='rgba(23, 156, 125)',
                              line_width=1,
                              layer='below')
                for i, hi in enumerate(hid):
                    if hi == 1:
                        hid_prev = hid.copy()
                        hid_prev[i] = 0
                        mean_start = hm_mapping[str(hid_prev)]
                        fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                                           ax=mean_start[0], ay=mean_start[1], x=mean[0], y=mean[1],
                                           showarrow=True, arrowhead=2, arrowsize=1.5)

            weights = list(weights[i, :] for i in range(weights.shape[0]))
            hid_states = [' '.join(list(hid_states[i, :].astype(int).astype(str))) for i in range(hid_states.shape[0])]
            # fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers',
            #                          hovertext=weights,
            #                          name='GMM',
            #                          textposition="top left", marker_color='orange'))
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            title_standoff=0,
            range=[ymin, ymax]
        )
        fig.update_xaxes(
            title_standoff=0,
            range=[xmin, xmax]
        )
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                          width=figure_width,
                          height=figure_height,
                          showlegend=showlegend,
                          legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01, orientation="h",
                                      font=dict(size=8)))
        fig.update_layout(**kwargs)


        return fig


class GaussianBinaryRBM (XBinaryRBM):
    def __init__(self, n_visible, n_hidden, device='cpu', replicate_h=1, sigma_learnable=False, sigma=1.):
        super().__init__(n_visible, n_hidden, device=device)

        if sigma_learnable:
            self.is_sigma_learnable = True
            self.log_sigma_x = nn.Parameter(np.log(sigma) * torch.ones(1, n_visible, requires_grad=True)).to(device)
        else:
            self.is_sigma_learnable = False
            self.log_sigma_x = np.log(sigma) * torch.ones(1, n_visible, requires_grad=False).to(device)
        self.weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01 * sigma).to(device)
        self.replicate_h = torch.tensor(replicate_h, requires_grad=False).float().to(device)


    def h2v(self, hidden):
        # sigma_x = torch.exp(self.log_sigma_x)
        x = torch.mul(self.replicate_h, torch.matmul(hidden, self.weights.t())) + self.visible_bias
        # x = torch.mul(x, sigma_x)
        return x

    def v2h(self, visible, **kwargs):
        sigma_x = torch.exp(self.log_sigma_x)
        if visible.dim() == 3:
            sigma_x = sigma_x.unsqueeze(2)
        visible = torch.div(visible, sigma_x.pow(2))
        return super().v2h(visible, **kwargs)

    def sample_x(self, visible_mean):
        """
        Sample visible states given hidden states.
        :param hidden: Input hidden states
        :return: Gaussian visible states
        """
        sigma_x = torch.exp(self.log_sigma_x)

        visible_states = visible_mean + sigma_x * torch.randn_like(visible_mean)  # Add Gaussian noise
        return visible_states

    def free_energy(self, x):
        x = x.view(x.size(0), -1)

        sigma = torch.exp(self.log_sigma_x)
        # Term 1: Quadratic term for visible units
        vb_term = torch.sum(((x - self.visible_bias) ** 2) / (2 * sigma ** 2), dim=1)

        # Term 2: Contribution from hidden units (using v2h to compute pre-sigmoid activations)
        hidden_activations = self.v2h(x, pre_sigmoid=True)
        hidden_term = self.replicate_h * torch.sum(F.softplus(hidden_activations), dim=1)  # log(1 + exp(...))

        # Combine terms
        return vb_term - hidden_term

    def gmm_model(self):
        # Transforms the p(v) into mixture of Gaussians and returns the weight, mean and sigma for each Gaussian component as
        # well the corresponding hidden states.This function is for use with very small models.Otherwize it will last forever

        with torch.no_grad():
            sigma = torch.exp(self.log_sigma_x).detach().numpy()
            num_components = 2 ** self.n_hidden
            gmm_sigmas = np.repeat(sigma, num_components, axis=0)

            # Initialize
            weights = np.zeros((num_components, 1))
            means = np.zeros((num_components, self.n_visible))
            hid_states = np.zeros((num_components, self.n_hidden))

            phi0 = np.prod(np.sqrt(2 * np.pi) * sigma)

            weights[0] = phi0
            means[0, :] = self.visible_bias.detach().numpy()
            for i in range(0, num_components):
                hs = list(bin(i)[2:])
                hid_states[i, -len(hs):] = hs
                hs = hid_states[i, :]
                # Calc means
                mean = self.h2v(torch.tensor(hs).float()).detach().numpy()
                means[i, :] = mean

                # Calc phi
                # phi = (np.sum(mean ** 2 / (2 * sigma ** 2)) - np.sum(self.visible_bias.detach().numpy() ** 2 / (2 * sigma ** 2)))
                # phi = np.sum(phi) + np.sum(self.hidden_bias.detach().numpy() * hs)
                # try:
                #     phi = phi0 * np.exp(phi)
                # except OverflowError:
                #     phi = np.iinfo(np.int64).max


                weights[i] = torch.exp(-self.free_energy(torch.tensor(mean)[None]))

            # Normalize weights
            try:
                Z = sum(weights)
            except OverflowError:
                Z = np.iinfo(np.int64).max
            weights = weights / Z
            return weights, means, gmm_sigmas, hid_states, Z



class ReLUBinaryRBM(XBinaryRBM):
    def __init__(self, n_visible, n_hidden, device='cpu', replicate_h=1):
        """
        Initialize the ReLU-Binary RBM.
        :param n_visible: Number of visible units
        :param n_hidden: Number of hidden units
        :param device: Device for computation (e.g., 'cpu', 'cuda')
        :param replicate_h: Factor to replicate hidden units' influence
        """
        super().__init__(n_visible, n_hidden, device=device)

        self.weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01).to(device)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible)).to(device)
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden)).to(device)
        self.replicate_h = torch.tensor(replicate_h, requires_grad=False).float().to(device)

    def h2v(self, hidden):
        """
        Hidden-to-Visible transformation using ReLU visible units.
        """
        # Compute ReLU activations for visible units
        visible_mean = torch.matmul(hidden, self.weights.t()) + self.visible_bias
        visible_states = F.relu(visible_mean)  # Ensure non-negativity
        return visible_states

    def v2h(self, visible, **kwargs):
        """
        Visible-to-Hidden transformation.
        """
        # Standard visible-to-hidden transformation
        pre_sigmoid_h = torch.matmul(visible, self.weights) + self.hidden_bias
        hidden_probs = torch.sigmoid(pre_sigmoid_h)
        hidden_states = torch.bernoulli(hidden_probs)
        return hidden_states, hidden_probs

    def free_energy(self, visible):
        """
        Compute the free energy for a given visible state.
        :param visible: Input visible state
        :return: Free energy
        """
        vb_term = 0.5 * torch.sum(visible ** 2, dim=1) - torch.sum(visible * self.visible_bias, dim=1)
        pre_sigmoid_h = torch.matmul(visible, self.weights) + self.hidden_bias
        hidden_term = torch.sum(F.softplus(pre_sigmoid_h), dim=1)  # log(1 + exp(...))
        return vb_term - hidden_term


class BinaryBinaryRBM (XBinaryRBM):
    def h2v(self, h):
        x = torch.matmul(h, self.weights.t()) + self.visible_bias
        return torch.sigmoid(x)

    def sample_x(self, vis_probs):
        """
        Sample hidden states given visible states.
        :param visible: Input visible states
        :return: Binary hidden states and probabilities
        """
        hidden_states = torch.bernoulli(vis_probs)
        return hidden_states

    def free_energy(self, x):
        """
        Compute the free energy of the visible units.
        :param x: Visible states
        :return: Free energy scalar
        """
        x = x.view(x.size(0), -1)
        vb_term = torch.matmul(x, self.visible_bias)
        wx_b = torch.matmul(x, self.weights) + self.hidden_bias
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vb_term - hidden_term


class PoissonBinaryRBM(XBinaryRBM):
    def __init__(self, n_visible, n_hidden, device='cpu', replicate_h=1):
        """
        Initialize the Poisson-Binary RBM.
        :param n_visible: Number of visible units
        :param n_hidden: Number of hidden units
        :param device: Device for computation (e.g., 'cpu', 'cuda')
        :param replicate_h: Factor to replicate hidden units' influence
        """
        super().__init__(n_visible, n_hidden, device=device)

        self.weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01).to(device)
        self.replicate_h = torch.tensor(replicate_h, requires_grad=False).float().to(device)

    def h2v(self, hidden):
        """
        Hidden-to-Visible transformation (mean of the Poisson distribution).
        """
        # Compute Poisson mean (rate parameter)
        poisson_mean = torch.mul(self.replicate_h, torch.matmul(hidden, self.weights.t())) + self.visible_bias
        poisson_mean = F.softplus(poisson_mean)  # Ensure positivity for Poisson mean
        return poisson_mean

    def v2h(self, visible, **kwargs):
        """
        Visible-to-Hidden transformation.
        """
        # Poisson-distributed visible units do not require variance scaling
        return super().v2h(visible, **kwargs)

    def sample_x(self, visible_mean):
        """
        Sample visible states given hidden states.
        :param visible_mean: Poisson mean (rate parameter)
        :return: Sampled Poisson visible states
        """
        # Sample from Poisson distribution
        visible_states = torch.poisson(visible_mean)
        return visible_states

    def free_energy(self, x):
        """
        Compute the free energy for a given visible state.
        :param x: Input visible state
        :return: Free energy
        """
        x = x.view(x.size(0), -1)

        # Term 1: Poisson negative log-likelihood
        poisson_term = -torch.sum(x * self.visible_bias - F.softplus(self.visible_bias), dim=1)

        # Term 2: Contribution from hidden units (using v2h to compute pre-sigmoid activations)
        hidden_activations = self.v2h(x, pre_sigmoid=True)
        hidden_term = self.replicate_h * torch.sum(F.softplus(hidden_activations), dim=1)  # log(1 + exp(...))

        # Combine terms
        return poisson_term - hidden_term


class ReluBinaryRBM (XBinaryRBM):
    def sample_x(self, hidden):
        """
        Sample visible states given hidden states.
        :param hidden: Input hidden states
        :return: Gaussian visible states
        """
        visible_mean = torch.matmul(hidden, self.weights.t()) + self.visible_bias
        visible_states = visible_mean + torch.randn_like(visible_mean)  # Add Gaussian noise
        return visible_states, visible_mean

    def free_energy(self, x):
        """
        Compute the free energy of the visible units.
        :param x: Visible states
        :return: Free energy scalar
        """
        vb_term = torch.matmul(x, self.visible_bias)
        wx_b = torch.matmul(x, self.weights) + self.hidden_bias
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vb_term - hidden_term


def mse( x, r, per_point=False):
    if per_point:
        return torch.mean(torch.square(x - r), dim=1)
    else:
        return torch.mean(torch.square(x - r))


def sparsity(activations, per_point=False):
    if per_point:
        return torch.mean(activations, dim=1)
    else:
        return torch.mean(activations)


class FractionalBinaryRBM (BinaryBinaryRBM):
    def __init__(self, n_visible, n_hidden, n_levels=5, device='cpu'):
        """
        Initialize the Fractional Binary RBM.
        :param n_visible: Number of visible units
        :param n_hidden: Number of hidden units
        :param device: Device for computation (e.g., 'cpu', 'cuda')
        :param n_levels: Number of fractional sub-phases
        """
        super().__init__(n_visible * n_levels, n_hidden, device)
        self.n_fractions = n_levels


    def decode_input(self, x):
        x = x.view(x.size(0), -1, self.n_fractions)
        x = x.sum(dim=2, keepdim=False)
        x = x * (self.max_values - self.min_values) / self.n_fractions + self.min_values
        return x


class ThreeWayBinaryBinaryRBM(BinaryBinaryRBM):
    """
    Three-Way Binary RBM with additional auxiliary/temporal units.
    """

    def __init__(self, n_visible, n_hidden, n_auxiliary, device='cpu'):
        """
        Initialize the Three-Way Binary RBM.
        :param n_visible: Number of visible units
        :param n_hidden: Number of hidden units
        :param n_auxiliary: Number of auxiliary/temporal units
        """
        super(ThreeWayBinaryBinaryRBM, self).__init__(n_visible, n_hidden, device=device)

        # Initialize three-way weights and biases
        self.three_way_weights = nn.Parameter(
            torch.randn(n_visible, n_hidden, n_auxiliary) * 0.01).to(device)
        self.auxiliary_bias = nn.Parameter(torch.zeros(n_auxiliary)).to(device)
        self.n_auxiliary = n_auxiliary

    def generate(self, num_examples, num_steps=10):
        with torch.no_grad():
            # Initialization
            # For v (visible units): Start with random binary values or use data input
            v = torch.randint(0, 2, (num_examples, self.n_visible)).float()

            # For u (third-layer units): Random binary values (e.g., Bernoulli distribution)
            u = torch.randint(0, 2, (num_examples, self.n_auxiliary)).float()

            for k in range(num_steps):
                ph = self.vu2h(v, u)
                h = self.sample_h(ph)
                pu = self.vh2u(v, h)
                u = self.sample_u(pu)
                px = self.hu2v(h, u)
                x = self.sample_x(px)
        return x

    def contrastive_divergence(self, d, num_gibbs=1, use_probability_last_x_update=False):
        """
        Perform Contrastive Divergence (CD-k) to train the RBM.
        :param visible: Input visible states
        :param num_gibbs: Number of Gibbs sampling steps
        :return: CD loss
        """

        with torch.no_grad():
            # Positive phase
            visible = d[:, :, -1]
            auxiliary = d[:, :, :-1]
            hidden_states = self.sample_h(self.vu2h(visible, auxiliary))
            auxiliary_states = auxiliary
            # Gibbs sampling
            for _ in range(num_gibbs):
                visible_states = self.sample_x(self.hu2v(hidden_states, auxiliary_states))
                auxiliary_states = self.sample_u(self.vh2u(visible_states, hidden_states))
                hidden_states = self.sample_h(self.vu2h(visible_states, auxiliary_states))

            # Negative phase
            negative_visible_states = self.hu2v(hidden_states, auxiliary_states)
            if not use_probability_last_x_update:
                negative_visible_states = self.sample_x(negative_visible_states)

        loss = torch.mean(self.free_energy(visible, auxiliary) - self.free_energy(negative_visible_states, auxiliary_states))
        return loss

    def vu2h(self, visible, auxiliary, pre_sigmoid=False):
        """
        Visible-to-Hidden transformation with three-way interactions.
        """
        visible = visible.view(visible.size(0), -1)
        auxiliary = auxiliary.reshape(auxiliary.size(0), -1)

        # Three-way contribution
        try:
            three_way_contrib = torch.einsum("bi,ijk,bk->bj", visible, self.three_way_weights, auxiliary)

            pre_sigmoid_h = torch.matmul(visible, self.weights) + self.hidden_bias + three_way_contrib
        except:
            print('y')
        if pre_sigmoid:
            return pre_sigmoid_h
        else:
            return torch.sigmoid(pre_sigmoid_h)

    def vh2u(self, v, h):
        # Compute activation for each u_k
        # Activation: sum over i (v_i * W_ijk) and j (h_j * W_ijk)
        activation = torch.einsum('bi,bj,ijk->bk', v, h, self.three_way_weights) + self.auxiliary_bias  # Efficient summation

        # Compute probabilities using sigmoid
        prob_u = torch.sigmoid(activation)
        return prob_u

    def sample_u(self, auxiliary_probs):
        """
        Sample auxiliary states given visible and hidden states.
        """
        auxiliary_states = torch.bernoulli(auxiliary_probs)
        return auxiliary_states

    def hu2v(self, hidden, auxiliary):
        """
        Hidden-to-Visible transformation with three-way interactions.
        """
        hidden = hidden.view(hidden.size(0), -1)
        auxiliary = auxiliary.view(auxiliary.size(0), -1)

        # Three-way contribution
        three_way_contrib = torch.einsum("bj,ijk,bk->bi", hidden, self.three_way_weights, auxiliary)
        visible_mean = torch.matmul(hidden, self.weights.t()) + self.visible_bias + three_way_contrib
        return torch.sigmoid(visible_mean)

    def a2v(self, auxiliary, hidden):
        """
        Auxiliary-to-Visible transformation (optional).
        """
        auxiliary = auxiliary.view(auxiliary.size(0), -1)
        hidden = hidden.view(hidden.size(0), -1)

        # Compute auxiliary-to-visible interactions
        three_way_contrib = torch.einsum("bk,ijk,bj->bi", auxiliary, self.three_way_weights, hidden)
        visible_mean = torch.matmul(hidden, self.weights.t()) + self.visible_bias + three_way_contrib
        return torch.sigmoid(visible_mean)


    def free_energy(self, visible, auxiliary):
        """
        Compute the free energy with three-way interactions.
        """
        visible = visible.view(visible.size(0), -1)
        auxiliary = auxiliary.view(auxiliary.size(0), -1)

        # Quadratic term for visible and auxiliary units
        vb_term = torch.sum(visible * self.visible_bias, dim=1)
        ab_term = torch.sum(auxiliary * self.auxiliary_bias, dim=1)

        # Hidden term with three-way contributions
        three_way_contrib = torch.einsum("bi,ijk,bk->bj", visible, self.three_way_weights, auxiliary)
        pre_sigmoid_h = torch.matmul(visible, self.weights) + self.hidden_bias + three_way_contrib
        hidden_term = torch.sum(F.softplus(pre_sigmoid_h), dim=1)

        # Combine terms
        return -(vb_term + ab_term + hidden_term)

