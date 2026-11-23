"""Exact-energy and training checks for the contextual three-way RBM."""

from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ml4cps.debta import DEBTA
from ml4cps.debta.rbms.bb import BinaryBinaryRBM
from ml4cps.debta.rbms.three_way import ThreeWayBinaryBinaryRBM
from ml4cps.debta.rbms.training import train_3way_rbm, train_rbm


class ThreeWayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ThreeWayBinaryBinaryRBM(2, 3, 2)
        self.v = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
        self.u = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        self.packed = torch.cat((self.v, self.u), -1)

    def test_inherits_binary_rbm_and_reduces_to_it_at_zero_gates(self):
        self.assertIsInstance(self.model, BinaryBinaryRBM)
        reference = BinaryBinaryRBM(2, 3)
        with torch.no_grad():
            reference.weights.copy_(self.model.weights)
            reference.visible_bias.copy_(self.model.visible_bias)
            reference.hidden_bias.copy_(self.model.hidden_bias)
            self.model.three_way_weights.zero_()
        torch.testing.assert_close(self.model.vu2h(self.v, self.u), reference.v2h(self.v))
        torch.testing.assert_close(self.model.hu2v(torch.ones(4, 3), self.u), reference.h2v(torch.ones(4, 3)))
        torch.testing.assert_close(self.model.free_energy(self.v, self.u), reference.free_energy(self.v))

    def test_free_energy_matches_exact_hidden_enumeration(self):
        model = self.model.double()
        v, u = self.v.double(), self.u.double()
        with torch.no_grad():
            model.visible_bias.copy_(torch.tensor([.2, -.4]))
            model.hidden_bias.copy_(torch.tensor([.3, -.1, .2]))
            model.auxiliary_bias.copy_(torch.tensor([-.5, .4]))
            model.three_way_weights.mul_(15.)
        hidden = torch.tensor(list(product((0., 1.), repeat=3)), dtype=torch.float64)
        energy = (-v @ model.visible_bias[:, None] - hidden @ model.hidden_bias
                  - (u @ model.auxiliary_bias)[:, None] - v @ model.weights @ hidden.T
                  - torch.einsum('bi,ija,kj,ba->bk', v, model.three_way_weights, hidden, u))
        torch.testing.assert_close(model.free_energy(v, u), -torch.logsumexp(-energy, dim=1))

    def test_gibbs_and_cd_hold_auxiliary_units_fixed(self):
        with patch.object(self.model, 'sample_u', side_effect=AssertionError('context was sampled')):
            result = self.model.gibbs_chain(self.packed, num_gibbs=3)
            torch.testing.assert_close(result[:, 2:], self.u)
            loss = self.model.contrastive_divergence(self.packed, num_gibbs=3)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.model.three_way_weights.grad).all())
        torch.testing.assert_close(self.model.auxiliary_bias.grad, torch.zeros(2))
        generated, auxiliary = self.model.generate(4, num_steps=2, auxiliary=self.u)
        self.assertEqual(generated.shape, (4, 2))
        torch.testing.assert_close(auxiliary, self.u)

    def test_legacy_layout_packed_reconstruction_and_temporal_regularization(self):
        legacy = torch.stack((self.u, self.v), dim=-1)
        torch.testing.assert_close(self.model.prepare_input(legacy), self.packed)
        torch.testing.assert_close(self.model.free_energy(legacy), self.model.free_energy(self.packed))
        reconstruction, hidden = self.model.recon(self.packed, round_hidden=True)
        self.assertEqual(reconstruction.shape, self.packed.shape)
        self.assertEqual(hidden.shape, (4, 3))
        torch.testing.assert_close(reconstruction[:, 2:], self.u)
        torch.testing.assert_close(hidden, hidden.round())
        window = self.packed.reshape(2, 2, 4)
        loss = self.model.compute_loss(window, num_gibbs=2, sparsity_weight=.01,
                                       temporal_persistence_weight=.01)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_persistent_chain_uses_each_new_minibatch_context(self):
        self.model.pcd_loss(self.packed, num_gibbs=2)
        changed = self.packed.clone()
        changed[:, 2:] = 1 - self.u
        loss = self.model.pcd_loss(changed, num_gibbs=2)
        torch.testing.assert_close(self.model.persistent_visible[:, 2:], changed[:, 2:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_native_checkpoint_reconstructs_auxiliary_dimension_and_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'three_way.pt'
            self.model.save(filename)
            restored = ThreeWayBinaryBinaryRBM.load(filename)
        self.assertIsInstance(restored, ThreeWayBinaryBinaryRBM)
        self.assertEqual(restored.n_auxiliary, 2)
        torch.testing.assert_close(restored.free_energy(self.packed), self.model.free_energy(self.packed))

    def test_native_trainers_accept_packed_and_legacy_inputs(self):
        with patch('plotly.graph_objects.Figure.show', return_value=None):
            train_rbm(self.model, self.packed, max_epoch=1, batch_size=2,
                      use_probability_last_v_update=False, verbose=False)
            self.assertEqual(self.model.num_epoch, 1)
            legacy = torch.stack((self.u, self.v), dim=-1)
            train_3way_rbm(self.model, legacy, max_epoch=1, batch_size=2,
                           use_probability_last_v_update=False, verbose=False)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in self.model.parameters()))

    def test_invalid_auxiliary_dimensions_layout_and_sampling_budget_are_rejected(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ThreeWayBinaryBinaryRBM(2, 3, size)
        with self.assertRaises(ValueError):
            self.model.prepare_input(torch.zeros(4, 3))
        with self.assertRaises(ValueError):
            self.model.effective_weights(torch.zeros(4, 3))
        with self.assertRaises(ValueError):
            self.model.gibbs_chain(self.packed, num_gibbs=0)

    def test_debta_temporal_model_keeps_sensor_latent_width(self):
        model = DEBTA(num_y=2, num_h=2, timed_model='rbm', log_mlflow=False)
        self.assertEqual(model.ebm.num_hidden_layers(), 1)
        self.assertEqual(model.ebm.timed_model.n_visible, 2)
        self.assertEqual(model.ebm.timed_model.n_auxiliary, 2)
        reconstructed, codes = model.recon(torch.zeros(4, 1, 2))
        self.assertEqual(reconstructed.shape, (4, 1, 2))
        self.assertEqual(codes.shape, (4, 1, 2))


class FactoredThreeWayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.model = ThreeWayBinaryBinaryRBM(2, 3, 2, n_factors=4).double()
        self.dense = ThreeWayBinaryBinaryRBM(2, 3, 2).double()
        self.v = torch.tensor([[0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
        self.u = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
        with torch.no_grad():
            for name in ('weights', 'visible_bias', 'hidden_bias', 'auxiliary_bias'):
                getattr(self.dense, name).copy_(getattr(self.model, name))
            self.dense.three_way_weights.copy_(torch.einsum(
                'if,jf,af->ija', self.model.visible_factors,
                self.model.hidden_factors, self.model.auxiliary_factors))

    def test_factored_model_matches_equivalent_full_tensor(self):
        self.assertFalse(hasattr(self.model, 'three_way_weights'))
        h = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.]], dtype=torch.float64)
        torch.testing.assert_close(self.model.effective_weights(self.u), self.dense.effective_weights(self.u))
        torch.testing.assert_close(self.model.three_way_hidden(self.v, self.u),
                                   self.dense.three_way_hidden(self.v, self.u))
        torch.testing.assert_close(self.model.vu2h(self.v, self.u), self.dense.vu2h(self.v, self.u))
        torch.testing.assert_close(self.model.hu2v(h, self.u), self.dense.hu2v(h, self.u))
        torch.testing.assert_close(self.model.vh2u(self.v, h), self.dense.vh2u(self.v, h))
        torch.testing.assert_close(self.model.free_energy(self.v, self.u), self.dense.free_energy(self.v, self.u))

    def test_energy_and_factor_gradients_match_explicit_hidden_enumeration(self):
        model = self.model
        hidden = torch.tensor(list(product((0., 1.), repeat=3)), dtype=torch.float64)
        interaction = ((self.v @ model.visible_factors)[:, None, :]
                       * (self.u @ model.auxiliary_factors)[:, None, :]
                       * (hidden @ model.hidden_factors)[None, :, :]).sum(-1)
        energy = (-self.v @ model.visible_bias[:, None] - hidden @ model.hidden_bias
                  - (self.u @ model.auxiliary_bias)[:, None]
                  - self.v @ model.weights @ hidden.T - interaction)
        exact = -torch.logsumexp(-energy, dim=1)
        actual = model.free_energy(self.v, self.u)
        torch.testing.assert_close(actual, exact)
        parameters = tuple(model.parameters())
        actual_grad = torch.autograd.grad(actual.sum(), parameters)
        exact_grad = torch.autograd.grad(exact.sum(), parameters)
        for left, right in zip(actual_grad, exact_grad):
            torch.testing.assert_close(left, right)
            self.assertTrue(torch.isfinite(left).all())

    def test_cd_and_pcd_keep_context_clamped_in_factored_model(self):
        packed = torch.cat((self.v, self.u), -1)
        generated = self.model.gibbs_chain(packed, num_gibbs=2)
        torch.testing.assert_close(generated[:, 2:], self.u)
        loss = self.model.contrastive_divergence(packed, num_gibbs=2)
        loss.backward()
        for factor in (self.model.visible_factors, self.model.hidden_factors, self.model.auxiliary_factors):
            self.assertIsNotNone(factor.grad)
            self.assertTrue(torch.isfinite(factor.grad).all())
        self.model.pcd_loss(packed, num_gibbs=1)
        changed = torch.cat((self.v, 1 - self.u), -1)
        self.model.pcd_loss(changed, num_gibbs=1)
        torch.testing.assert_close(self.model.persistent_visible[:, 2:], 1 - self.u)

    def test_checkpoint_restores_factored_model_without_unused_bias_or_tensor(self):
        model = ThreeWayBinaryBinaryRBM(2, 3, 2, n_factors=4, auxiliary_bias=False).double()
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'factored.pt'
            model.save(filename)
            restored = ThreeWayBinaryBinaryRBM.load(filename)
        self.assertEqual(restored.n_factors, 4)
        self.assertIsNone(restored.auxiliary_bias)
        self.assertFalse(hasattr(restored, 'three_way_weights'))
        self.assertEqual(restored.visible_factors.dtype, torch.float64)
        torch.testing.assert_close(restored.free_energy(self.v, self.u), model.free_energy(self.v, self.u))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(dict(restored.named_parameters())[name], parameter)

    def test_invalid_factor_counts_and_auxiliary_bias_option(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ThreeWayBinaryBinaryRBM(2, 3, 2, n_factors=value)
        with self.assertRaises(ValueError):
            ThreeWayBinaryBinaryRBM(2, 3, 2, auxiliary_bias=1)





class MixedAuxiliaryTests(unittest.TestCase):
    def test_mixed_free_energy_and_gradients_match_hidden_enumeration(self):
        hidden = torch.tensor(list(product((0., 1.), repeat=3)), dtype=torch.float64)
        visible = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
        auxiliary = torch.tensor([[0., -1.4], [1., 2.1]], dtype=torch.float64)
        for factors in (None, 4):
            with self.subTest(factors=factors):
                model = ThreeWayBinaryBinaryRBM(2, 3, 2, n_factors=factors,
                    auxiliary_types=('binary', 'gaussian'), auxiliary_sigma=[1., .7]).double()
                with torch.no_grad():
                    model.auxiliary_bias.copy_(torch.tensor([.2, -.3]))
                tensor = model.three_way_weights if factors is None else torch.einsum(
                    'if,jf,af->ija', model.visible_factors, model.hidden_factors, model.auxiliary_factors)
                sigma = model.auxiliary_sigma
                auxiliary_energy = (-auxiliary[:, 0] * model.auxiliary_bias[0]
                    + .5 * ((auxiliary[:, 1] - model.auxiliary_bias[1]) / sigma[1]).square())
                energy = (-visible @ model.visible_bias[:, None] - hidden @ model.hidden_bias
                    - visible @ model.weights @ hidden.T + auxiliary_energy[:, None]
                    - torch.einsum('bi,ija,kj,ba->bk', visible, tensor, hidden, auxiliary / sigma))
                expected = -torch.logsumexp(-energy, dim=1)
                actual = model.free_energy(visible, auxiliary)
                torch.testing.assert_close(actual, expected)
                parameters = tuple(model.parameters())
                for actual_grad, expected_grad in zip(torch.autograd.grad(actual.sum(), parameters),
                                                      torch.autograd.grad(expected.sum(), parameters)):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_mixed_auxiliary_conditionals_sample_gaussian_mean_and_variance(self):
        torch.manual_seed(172)
        model = ThreeWayBinaryBinaryRBM(1, 1, 2,
            auxiliary_types=['binary', 'gaussian'], auxiliary_sigma=[1., 2.]).double()
        with torch.no_grad():
            model.three_way_weights.copy_(torch.tensor([[[.4, -.6]]]))
            model.auxiliary_bias.copy_(torch.tensor([.2, .3]))
        moments = model.vh2u(torch.ones(1, 1), torch.ones(1, 1))
        expected = torch.tensor([[torch.sigmoid(torch.tensor(.6)).item(), .3 + 2. * -.6]], dtype=torch.float64)
        torch.testing.assert_close(moments, expected)
        samples = model.sample_u(moments.expand(40000, -1))
        self.assertTrue(((samples[:, 0] == 0) | (samples[:, 0] == 1)).all())
        self.assertAlmostEqual(float(samples[:, 0].mean()), float(moments[0, 0]), delta=.012)
        self.assertAlmostEqual(float(samples[:, 1].mean()), float(moments[0, 1]), delta=.035)
        self.assertAlmostEqual(float(samples[:, 1].var()), 4., delta=.12)

    def test_mixed_auxiliary_stays_clamped_and_survives_native_checkpoint(self):
        model = ThreeWayBinaryBinaryRBM(2, 3, 2, n_factors=3,
            auxiliary_types=['binary', 'gaussian'], auxiliary_sigma=[1., 2.]).double()
        packed = torch.tensor([[0., 1., 0., -2.4], [1., 0., 1., 1.3]], dtype=torch.float64)
        torch.testing.assert_close(model.gibbs_chain(packed, num_gibbs=3)[:, 2:], packed[:, 2:])
        loss = model.contrastive_divergence(packed, num_gibbs=2)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mixed.pt'
            model.save(path)
            restored = ThreeWayBinaryBinaryRBM.load(path)
        self.assertEqual(restored.auxiliary_types, ('binary', 'gaussian'))
        torch.testing.assert_close(restored.auxiliary_sigma, model.auxiliary_sigma)
        torch.testing.assert_close(restored.free_energy(packed), model.free_energy(packed))

    def test_mixed_auxiliary_validation_and_legacy_state_defaults(self):
        for types, sigma in ((['binary'], 1.), (['binary', 'other'], 1.),
                             (['binary', 'gaussian'], [2., 1.]),
                             (['binary', 'gaussian'], [1., 0.]),
                             (['binary', 'gaussian'], float('nan'))):
            with self.subTest(types=types, sigma=sigma), self.assertRaises(ValueError):
                ThreeWayBinaryBinaryRBM(2, 3, 2, auxiliary_types=types, auxiliary_sigma=sigma)
        model = ThreeWayBinaryBinaryRBM(2, 3, 2,
            auxiliary_types=['binary', 'gaussian'], auxiliary_sigma=2.)
        torch.testing.assert_close(model.auxiliary_sigma, torch.tensor([1., 2.]))
        for values in ([[.3, 1.]], [[1., float('inf')]], [[0., float('nan')]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.effective_weights(values)
        binary = ThreeWayBinaryBinaryRBM(2, 3, 2)
        legacy = binary.state_dict()
        del legacy['_gaussian_auxiliary'], legacy['auxiliary_sigma']
        model.load_state_dict(legacy)
        self.assertEqual(model.auxiliary_types, ('binary', 'binary'))
        torch.testing.assert_close(model.auxiliary_sigma, torch.ones(2))


if __name__ == '__main__':
    unittest.main()
