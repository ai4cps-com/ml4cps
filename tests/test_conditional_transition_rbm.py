"""Conditional transitions share native RBM machinery without losing support."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ml4cps.debta.debta_tools import transition_counts, transition_vector
from ml4cps.debta.rbms.bb import BinaryBinaryRBM, ConditionalTransitionRBM


class ConditionalTransitionTests(unittest.TestCase):
    def setUp(self):
        self.model = ConditionalTransitionRBM(1, [[0, 0], [0, 1], [1, 0]],
                                              n_hidden=4, n_factors=3, seed=13)
        self.source = torch.tensor([[1., 0.]])
        self.configuration = torch.tensor([[1.]])

    def test_native_parameters_and_conditional_energy_are_differentiable(self):
        self.assertIsInstance(self.model, BinaryBinaryRBM)
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


if __name__ == '__main__':
    unittest.main()
