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


if __name__ == '__main__':
    unittest.main()
