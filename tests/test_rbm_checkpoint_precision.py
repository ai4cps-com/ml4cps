"""Checkpoints must not round high-precision parameters on reconstruction."""

from pathlib import Path
import tempfile
import unittest

import torch

from ml4cps.debta.rbms.bb import BinaryBinaryRBM
from ml4cps.debta.rbms.gb import GaussianBinaryRBM
from ml4cps.debta.rbms.three_way import ThreeWayBinaryBinaryRBM


class CheckpointPrecisionTests(unittest.TestCase):
    def test_binary_gaussian_and_three_way_double_parameters_roundtrip_exactly(self):
        models = (BinaryBinaryRBM(2, 3), GaussianBinaryRBM(2, 3), ThreeWayBinaryBinaryRBM(2, 3, 2))
        for model in models:
            with self.subTest(model=type(model).__name__), tempfile.TemporaryDirectory() as folder:
                model.double()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(.123456789123456)
                filename = Path(folder) / 'double.pt'
                model.save(filename)
                restored = type(model).load(filename)
                self.assertEqual(restored.weights.dtype, torch.float64)
                for name, expected in model.state_dict().items():
                    actual = restored.state_dict()[name]
                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gaussian_buffer_can_keep_a_different_dtype_from_its_parameters(self):
        model = GaussianBinaryRBM(2, 3).double()
        model.log_sigma_v = model.log_sigma_v.float()
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'mixed.pt'
            model.save(filename)
            restored = GaussianBinaryRBM.load(filename)
        self.assertEqual(restored.weights.dtype, torch.float64)
        self.assertEqual(restored.log_sigma_v.dtype, torch.float32)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
