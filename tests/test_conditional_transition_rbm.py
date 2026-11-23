"""Conditional transitions share native RBM machinery without losing support."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ml4cps.debta.debta_tools import transition_counts, transition_vector
from ml4cps.debta.rbms.bb import BinaryBinaryRBM
from ml4cps.debta.rbms.three_way import ConditionalTransitionRBM, ThreeWayBinaryBinaryRBM


class ConditionalTransitionTests(unittest.TestCase):
    def setUp(self):
        self.model = ConditionalTransitionRBM(1, [[0, 0], [0, 1], [1, 0]],
                                              n_hidden=4, n_factors=3, seed=13)
        self.source = torch.tensor([[1., 0.]])
        self.configuration = torch.tensor([[1.]])

    def test_native_parameters_and_conditional_energy_are_differentiable(self):
        self.assertIsInstance(self.model, BinaryBinaryRBM)
        self.assertIsInstance(self.model, ThreeWayBinaryBinaryRBM)
        self.assertIs(self.model.transition_parameters['G'], self.model.auxiliary_factors)
        self.assertIs(self.model.transition_parameters['Ud'], self.model.visible_factors)
        self.assertIs(self.model.transition_parameters['V'], self.model.hidden_factors)
        self.assertIsNone(self.model.auxiliary_bias)
        self.assertFalse(hasattr(self.model, 'three_way_weights'))
        parameters = list(self.model.parameters())
        self.assertTrue(parameters)
        self.assertTrue(all(isinstance(value, torch.nn.Parameter) for value in parameters))
        self.assertEqual({id(p) for p in parameters},
                         {id(p) for p in self.model.transition_parameters.values()})
        self.assertIs(self.model.weights, self.model.transition_parameters['Wd'])
        visible = torch.tensor([[0., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        energy = self.model.free_energy(visible, source=self.source,
                                       configuration=self.configuration)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertTrue(energy.requires_grad)
        energy.sum().backward()
        self.assertTrue(torch.isfinite(visible.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))

    def test_float32_inputs_work_with_default_float64_model(self):
        visible = torch.tensor([[0., 1.]], dtype=torch.float32)
        hidden = self.model.v2h(visible, source=self.source, configuration=self.configuration)
        probabilities = self.model.h2v(hidden.float(), source=self.source,
                                       configuration=self.configuration)
        self.assertEqual(hidden.dtype, torch.float64)
        self.assertEqual(probabilities.dtype, torch.float64)
        self.assertEqual(tuple(probabilities.shape), (1, 3))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1, dtype=torch.float64))

    def test_conditional_gibbs_step_and_chain_never_emit_forbidden_codes(self):
        torch.manual_seed(42)
        with torch.no_grad():
            # Independent Bernoulli sampling would overwhelmingly choose 11.
            self.model.visible_bias.fill_(12.)
        initial = torch.zeros(512, 2)
        state, probability, hidden_state, hidden_probability = self.model.gibbs_step(
            initial, source=self.source, configuration=self.configuration)
        self.assertEqual(tuple(probability.shape), (512, 3))
        self.assertEqual(tuple(hidden_state.shape), (512, 4))
        self.assertEqual(tuple(hidden_probability.shape), (512, 4))
        self.assertTrue(torch.isin(hidden_state, torch.tensor([0., 1.], dtype=hidden_state.dtype)).all())
        for values in (state, self.model.gibbs_chain(
                initial, num_gibbs=5, source=self.source, configuration=self.configuration)):
            self.assertTrue(((values[:, None] == self.model.state_codes[None]).all(-1).any(-1)).all())
            self.assertFalse((values == 1).all(-1).any())

    def test_pt_and_npz_roundtrips_preserve_parameters_and_fit_metadata(self):
        self.model.fit_info = {'success': True, 'objective': 'conditional NLL', 'value': .125}
        expected = self.model.log_probabilities([[0], [1]])
        with tempfile.TemporaryDirectory() as folder:
            for suffix in ('.pt', '.npz'):
                with self.subTest(suffix=suffix):
                    filename = Path(folder) / ('conditional' + suffix)
                    self.model.save(filename)
                    restored = ConditionalTransitionRBM.load(filename)
                    self.assertEqual(restored.fit_info, self.model.fit_info)
                    np.testing.assert_array_equal(restored.log_probabilities([[0], [1]]), expected)
                    torch.testing.assert_close(restored.state_codes, self.model.state_codes)

    def test_genuine_numpy_object_checkpoint_migrates_to_native_parameters(self):
        fixture = Path(__file__).parent / 'data' / 'conditional_transition_numpy.pt'
        checkpoint = torch.load(fixture, map_location='cpu', weights_only=False)
        restored = checkpoint['model']
        self.assertIsInstance(restored, ConditionalTransitionRBM)
        self.assertIsInstance(restored.weights, torch.nn.Parameter)
        self.assertTrue(all(isinstance(value, torch.nn.Parameter) for value in restored.parameters()))
        self.assertEqual(restored.fit_info, checkpoint['fit_info'])
        np.testing.assert_allclose(restored.log_probabilities([[0], [1]]),
                                   checkpoint['log_probability'], rtol=1e-12, atol=1e-12)

    def test_old_native_and_object_checkpoints_load_in_a_fresh_process(self):
        script = """
import sys
import numpy as np
import torch

assert not any(k == 'ml4cps' or k.startswith('ml4cps.') for k in sys.modules)
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
from ml4cps.debta.rbms.base import XBinaryRBM
from ml4cps.debta.rbms.three_way import ConditionalTransitionRBM, ThreeWayBinaryBinaryRBM
from ml4cps.debta.rbms.bb import ConditionalTransitionRBM as PreviousPath
assert PreviousPath is ConditionalTransitionRBM
models = [checkpoint['model'], XBinaryRBM.load(sys.argv[2])]
for model in models:
    assert isinstance(model, ThreeWayBinaryBinaryRBM)
    assert model.n_auxiliary == model.n_config
    assert model.auxiliary_bias is None
    assert not hasattr(model, 'three_way_weights')
    assert 'G' not in model.context_weights
    assert {id(p) for p in model.parameters()} == {id(p) for p in model.transition_parameters.values()}
    np.testing.assert_allclose(model.log_probabilities([[0], [1]]), checkpoint['expected'], rtol=1e-13, atol=1e-13)
    assert model.fit_info['objective'] == 'compatibility fixture'
"""
        fixtures = Path(__file__).parent / 'data'
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONPATH'] = os.pathsep.join(str(Path(p or '.').resolve()) for p in sys.path)
        result = subprocess.run([sys.executable, '-c', script,
                                 str(fixtures / 'conditional_transition_object.pt'),
                                 str(fixtures / 'conditional_transition_native.pt')],
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generation_requires_both_contexts_and_stays_in_the_codebook(self):
        with self.assertRaises(TypeError):
            self.model.generate(8)
        with self.assertRaises(ValueError):
            self.model.generate(0, source=self.source, configuration=self.configuration)
        values = self.model.generate(128, num_steps=3, source=self.source,
                                     configuration=self.configuration)
        self.assertEqual(tuple(values.shape), (128, 2))
        self.assertTrue(((values[:, None] == self.model.state_codes[None]).all(-1).any(-1)).all())

    def test_transition_vector_rejects_noninteger_indices(self):
        for source, destination in ((1., 0), (0, 1.), (-1, 0), (0, 3), (True, 0)):
            with self.subTest(source=source, destination=destination):
                with self.assertRaises(ValueError):
                    transition_vector([1], source, destination, [[0, 0], [0, 1], [1, 0]])
        np.testing.assert_array_equal(
            transition_vector([1], np.int64(1), 2, [[0, 0], [0, 1], [1, 0]]), [1, 0, 1, 1, 0])

    def test_transition_counts_rejects_invalid_modes_instead_of_casting_or_wrapping(self):
        frame = pd.DataFrame({'sequence_id': ['s'] * 3, 'time_s': [0., 1., 2.],
                              'configuration': ['0'] * 3, 'mode': [0., 1., 0.]})
        np.testing.assert_array_equal(transition_counts(frame, ['0'], 2, 1.),
                                      [[[0, 1], [1, 0]]])
        for invalid in (-1., .5, 2., np.nan):
            with self.subTest(mode=invalid):
                broken = frame.copy()
                broken.loc[1, 'mode'] = invalid
                with self.assertRaises(ValueError):
                    transition_counts(broken, ['0'], 2, 1.)





class MixedConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.model = ConditionalTransitionRBM(2, [[0, 0], [0, 1], [1, 0]],
            n_hidden=3, n_factors=2, seed=14,
            configuration_types=('binary', 'gaussian'), configuration_sigma=[1., .7])

    def test_mixed_conditioned_energy_and_gradients_match_explicit_equation(self):
        model = self.model
        torch.manual_seed(7)
        with torch.no_grad():
            model.context_weights['Ac'].normal_(std=.3)
            model.context_weights['Bc'].normal_(std=.3)
        p = model.transition_parameters
        config = torch.tensor([[1., -2.3]], dtype=torch.float64, requires_grad=True)
        source = torch.tensor([[1., 0.]], dtype=torch.float64)
        hidden = torch.tensor(list(__import__('itertools').product((0., 1.), repeat=3)), dtype=torch.float64)
        scaled = config / model.configuration_sigma
        gate = scaled @ p['G']
        a = p['a'] + scaled @ p['Ac'] + source @ p['As']
        b = p['b'] + scaled @ p['Bc'] + source @ p['Ws'] + ((source @ p['Us']) * gate) @ p['V'].T
        weights = p['Wd'] + torch.einsum('bf,if,jf->bij', gate, p['Ud'], p['V'])[0]
        energy = (-model.state_codes @ a.T - (hidden @ b.T).T
                  - model.state_codes @ weights @ hidden.T)
        expected = -torch.logsumexp(-energy, dim=1)
        actual = model.free_energy(model.state_codes, source=source, configuration=config)
        torch.testing.assert_close(actual, expected)
        parameters = tuple(model.parameters()) + (config,)
        for left, right in zip(torch.autograd.grad(actual.sum(), parameters),
                               torch.autograd.grad(expected.sum(), parameters)):
            torch.testing.assert_close(left, right)
        np.testing.assert_allclose(np.exp(model.log_probabilities([[0., -1.4], [1., 2.1]])).sum(-1), 1.)
        self.assertEqual(float(model.configuration_sigma[1]), .7)

    def test_mixed_pt_npz_and_object_roundtrips_preserve_scales_and_probabilities(self):
        configurations = [[0., -1.3], [1., 2.6]]
        expected = self.model.log_probabilities(configurations)
        with tempfile.TemporaryDirectory() as folder:
            for suffix in ('.pt', '.npz', '.object'):
                with self.subTest(suffix=suffix):
                    path = Path(folder) / ('mixed' + suffix)
                    if suffix == '.object':
                        torch.save(self.model, path)
                        restored = torch.load(path, weights_only=False)
                    else:
                        self.model.save(path)
                        restored = ConditionalTransitionRBM.load(path)
                    self.assertEqual(restored.configuration_types, ('binary', 'gaussian'))
                    self.assertEqual(float(restored.configuration_sigma[1]), .7)
                    np.testing.assert_array_equal(restored.log_probabilities(configurations), expected)

    def test_real_configuration_fitting_and_transition_vectors(self):
        from ml4cps.debta.rbms.training import fit_transition_rbm
        counts = np.array([[[20, 1], [20, 1]], [[1, 20], [1, 20]]])
        model = fit_transition_rbm([[-1.5], [1.5]], counts, state_codes=[[0], [1]],
            n_hidden=2, n_factors=2, configuration_types=['gaussian'],
            configuration_sigma=.7, max_iter=150, l2=.0001)
        probabilities = np.exp(model.log_probabilities([[-1.5], [1.5]]))
        self.assertGreater(probabilities[0, 0, 0], .8)
        self.assertGreater(probabilities[1, 0, 1], .8)
        self.assertEqual(model.fit_info['configuration_types'], ['gaussian'])
        self.assertEqual(model.fit_info['configuration_bits'], 0)
        self.assertEqual(model.fit_info['configuration_sigma'], [.7])
        np.testing.assert_array_equal(transition_vector([1., -2.4], 0, 1, [[0], [1]],
            configuration_types=['binary', 'gaussian']), [1., -2.4, 0., 1.])
        with self.assertRaises(ValueError):
            transition_vector([1., -2.4], 0, 1, [[0], [1]])
        with self.assertRaises(ValueError):
            self.model.log_probabilities([[.5, 1.4]])
        with self.assertRaises(ValueError):
            self.model.log_probabilities([[1., float('inf')]])

    def test_auxiliary_reverse_moments_use_all_context_terms_and_require_source(self):
        model = self.model
        torch.manual_seed(11)
        with torch.no_grad():
            model.context_weights['Ac'].normal_(std=.3)
            model.context_weights['Bc'].normal_(std=.3)
        visible = torch.tensor([[0., 1.]], dtype=torch.float64)
        hidden = torch.tensor([[1., 0., 1.]], dtype=torch.float64)
        source = torch.tensor([[1., 0.]], dtype=torch.float64)
        with self.assertRaises(TypeError):
            model.vh2u(visible, hidden)
        moments = model.vh2u(visible, hidden, source=source)
        config = torch.tensor([[0., float(moments[0, 1])]], dtype=torch.float64, requires_grad=True)
        a, b, weights = model.conditional_parameters(config, source)
        raw = (-visible @ a.T - hidden @ b.T
               - torch.einsum('bi,bij,bj->b', visible, weights, hidden)
               + .5 * (config[:, 1] / model.configuration_sigma[1]).square())
        derivative = torch.autograd.grad(raw.sum(), config)[0]
        self.assertAlmostEqual(float(derivative[0, 1]), 0., places=12)


if __name__ == '__main__':
    unittest.main()
