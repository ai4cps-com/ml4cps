"""
Bernoulli-Bernoulli Restricted Boltzmann Machine.

This module defines :class:`BinaryBinaryRBM`, an RBM with Bernoulli visible
units and Bernoulli hidden units.

Author
------
Nemanja Hranisavljevic
hranisan@hsu.hamburg
nemanja@ai4cps.com
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ml4cps.debta.rbms.base import XBinaryRBM


class BinaryBinaryRBM(XBinaryRBM):
    """
    Bernoulli-Bernoulli Restricted Boltzmann Machine.

    In this model, both visible and hidden units are binary random variables.

    The conditional distributions are:

    .. math::
        p(h_j = 1 \mid v) = \sigma\left(\sum_i v_i W_{ij} + b^h_j\right)

    and

    .. math::
        p(v_i = 1 \mid h) = \sigma\left(\sum_j h_j W_{ij} + b^v_i\right)

    where:

    - ``W`` is the weight matrix of shape ``(n_visible, n_hidden)``
    - ``b^v`` is the visible bias vector
    - ``b^h`` is the hidden bias vector
    - ``sigma`` denotes the logistic sigmoid

    Notes
    -----
    This is the standard binary-visible, binary-hidden RBM. It inherits the
    hidden conditional :meth:`v2h`, Bernoulli hidden sampling, Gibbs updates,
    and generic training-loss helpers from :class:`XBinaryRBM`.
    """

    def h2v(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute visible Bernoulli probabilities given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden activations or sampled hidden states with shape
            ``(batch_size, n_hidden)``.

        Returns
        -------
        torch.Tensor
            Visible-unit Bernoulli probabilities with shape
            ``(batch_size, n_visible)``.
        """
        visible_logits = h @ self.weights.t() + self.visible_bias
        return torch.sigmoid(visible_logits)

    def sample_v(self, v_probs: torch.Tensor) -> torch.Tensor:
        """
        Sample binary visible states from Bernoulli probabilities.

        Parameters
        ----------
        v_probs : torch.Tensor
            Visible-unit Bernoulli probabilities.

        Returns
        -------
        torch.Tensor
            Binary visible samples with the same shape as ``v_probs``.

        Notes
        -----
        The input is clamped to ``[0, 1]`` before sampling for numerical
        robustness.
        """
        v_probs = torch.clamp(v_probs, 0.0, 1.0)
        return torch.bernoulli(v_probs)

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy for a batch of visible samples.

        Parameters
        ----------
        v : torch.Tensor
            Visible states of shape ``(batch_size, n_visible)`` or any tensor
            that can be flattened to that shape.

        Returns
        -------
        torch.Tensor
            Per-sample free-energy values with shape ``(batch_size,)``.

        Notes
        -----
        For a Bernoulli-Bernoulli RBM, the free energy is

        .. math::
            F(v) = - v^T b^v - \sum_j \operatorname{softplus}(v^T W_{:,j} + b^h_j)

        where ``softplus(x) = log(1 + exp(x))`` is used for numerical stability.
        """
        v = v.reshape(v.size(0), -1)
        visible_term = v @ self.visible_bias
        hidden_logits = v @ self.weights + self.hidden_bias
        hidden_term = torch.sum(F.softplus(hidden_logits), dim=1)
        return -visible_term - hidden_term