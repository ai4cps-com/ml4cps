"""Explicit Gaussian mixtures must represent the same density as the RBM."""

import math
import unittest

import numpy as np
import torch

from ml4cps.debta.rbms.gb import GaussianBinaryRBM


class GaussianMixtureTests(unittest.TestCase):
    def test_one_hidden_component_odds_match_completed_square(self):
        model = GaussianBinaryRBM(1, 1, sigma=.7).double()
        with torch.no_grad():
            model.log_sigma_v.fill_(math.log(.7))
            model.weights.fill_(1.2)
            model.hidden_bias.fill_(-.8)
            model.visible_bias.fill_(.4)
        weights, means, sigmas, hidden, partition = model.gmm_model()
        log_odds = -.8 + (.4 / .7) * 1.2 + .5 * 1.2 ** 2
        probability = 1 / (1 + math.exp(-log_odds))
        np.testing.assert_allclose(weights[:, 0], [1 - probability, probability], rtol=1e-12)
        np.testing.assert_allclose(means[:, 0], [.4, .4 + .7 * 1.2], rtol=1e-12)
        np.testing.assert_allclose(sigmas[:, 0], [.7, .7], rtol=1e-12)
        np.testing.assert_array_equal(hidden[:, 0], [0., 1.])
        self.assertAlmostEqual(partition, 1 + math.exp(log_odds), places=12)

    def test_mixture_and_normalized_free_energy_have_the_same_density(self):
        model = GaussianBinaryRBM(2, 2).double()
        with torch.no_grad():
            model.log_sigma_v.copy_(torch.tensor([[.3, .8]], dtype=torch.float64).log())
            model.weights.copy_(torch.tensor([[.2, .9], [-.7, .4]], dtype=torch.float64))
            model.hidden_bias.copy_(torch.tensor([.5, -.3], dtype=torch.float64))
            model.visible_bias.copy_(torch.tensor([.1, -.2], dtype=torch.float64))
        weights, means, sigmas, _, partition = model.gmm_model()
        points = torch.tensor([[-.3, 1.2], [.1, -.2], [1.5, .6]], dtype=torch.float64)
        normal = torch.distributions.Normal(torch.from_numpy(means), torch.from_numpy(sigmas))
        mixture_log_probability = torch.logsumexp(
            torch.from_numpy(weights[:, 0]).log()[None, :]
            + normal.log_prob(points[:, None, :]).sum(-1), dim=1)
        gaussian_integral_log = points.shape[1] / 2 * math.log(2 * math.pi) + model.log_sigma_v.sum()
        rbm_log_probability = -model.free_energy(points) - math.log(partition) - gaussian_integral_log
        torch.testing.assert_close(mixture_log_probability, rbm_log_probability, rtol=1e-12, atol=1e-12)

    def test_large_parameters_still_give_finite_normalized_weights(self):
        model = GaussianBinaryRBM(1, 1).double()
        with torch.no_grad():
            model.weights.fill_(50.)
            model.hidden_bias.zero_()
            model.visible_bias.zero_()
        weights, _, _, _, partition = model.gmm_model()
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertTrue(math.isinf(partition))


if __name__ == '__main__':
    unittest.main()
