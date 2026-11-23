"""Contracts for observation encoding through DEBTA's reconstruction API."""

import unittest
import warnings

import numpy as np
import torch

from ml4cps.debta import DEBTA


class EncodeObservationsTests(unittest.TestCase):
    def identity_model(self):
        model = DEBTA(num_y=3, num_h=3, sigma=.5, log_mlflow=False)
        with torch.no_grad():
            model.ebm._discretization_layers[0].weight.copy_(torch.eye(3))
            model.ebm._discretization_layers[0].bias.zero_()
            model.ebm._decoder_biases[0].zero_()
        return model

    def test_packs_binary_codes_and_restores_physical_sensor_units(self):
        model = self.identity_model()
        observations = np.array([[12., 95., 1010.], [8., 105., 1010.]])
        codes, reconstructed = model.encode_observations(observations, [10., 100., 1000.], [2., 5., 10.])
        np.testing.assert_array_equal(codes, [5, 6])
        np.testing.assert_allclose(reconstructed, [[11., 100., 1005.], [10., 102.5, 1005.]])
        self.assertEqual(codes.shape, (2,))
        self.assertEqual(reconstructed.shape, observations.shape)
        self.assertTrue(np.issubdtype(codes.dtype, np.integer))

    def test_matches_existing_rounded_reconstruction_for_multiple_layers(self):
        model = DEBTA(num_y=2, num_h=3, first_hidden_size=4, num_sigm_layers=2,
                      sigma=.3, log_mlflow=False)
        observations = np.array([[12., 30.], [16., 20.]], dtype=np.float32)
        mean, std = np.array([10., 20.], dtype=np.float32), np.array([2., 5.], dtype=np.float32)
        normalized = torch.as_tensor((observations - mean) / std)[:, None, :]
        with torch.no_grad():
            expected_reconstruction, expected_bits = model.recon(normalized)
        bits = expected_bits.cpu().numpy()[:, -1, :].astype(int)
        expected_codes = bits @ (2 ** np.arange(bits.shape[1]))
        codes, reconstructed = model.encode_observations(observations, mean, std)
        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_allclose(reconstructed, expected_reconstruction.cpu().numpy()[:, -1, :] * std + mean)

    def test_accepts_readonly_sensor_arrays_without_mutating_them(self):
        model = self.identity_model()
        observations = np.array([[12., 95., 1010.], [8., 105., 1010.]])
        before = observations.copy()
        observations.setflags(write=False)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            codes, _ = model.encode_observations(observations, [10., 100., 1000.], [2., 5., 10.])
        np.testing.assert_array_equal(observations, before)
        np.testing.assert_array_equal(codes, [5, 6])
        self.assertFalse(any('not writable' in str(item.message) for item in captured))

    def test_inference_preserves_weights_statistics_and_gradients(self):
        model = self.identity_model()
        before = {name: value.clone() for name, value in model.ebm.state_dict().items()}
        mean, std = np.array([10., 100., 1000.]), np.array([2., 5., 10.])
        model.encode_observations([[12., 95., 1010.]], mean, std)
        np.testing.assert_array_equal(mean, [10., 100., 1000.])
        np.testing.assert_array_equal(std, [2., 5., 10.])
        for name, value in model.ebm.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.ebm.parameters()))


if __name__ == '__main__':
    unittest.main()
