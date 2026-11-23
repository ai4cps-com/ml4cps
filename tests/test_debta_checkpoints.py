"""Native checkpoints preserve declared subclass state without overrides."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ml4cps.debta import DEBTA


class TaggedDEBTA(DEBTA):
    checkpoint_attributes = ("tag", "calibration")
    tag = "unfitted"
    calibration = None


class OtherDEBTA(DEBTA):
    pass


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = TaggedDEBTA(num_y=2, num_h=2, log_mlflow=False)
        self.model.tag = "trained"
        self.model.calibration = np.array([.5, 2.])
        self.model._mean = np.array([3., 4.])
        self.model._std = np.array([.2, .3])
        self.model._G.add_edge("a", "b", event="switch")

    def test_subclass_roundtrip_uses_one_read_and_preserves_core_state(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "tagged.pt"
            self.model.save(filename)
            original_load = torch.load
            with patch("ml4cps.debta.base.torch.load", wraps=original_load) as load:
                restored = TaggedDEBTA.load(filename)
            self.assertEqual(load.call_count, 1)
        self.assertIsInstance(restored, TaggedDEBTA)
        self.assertEqual(restored.tag, self.model.tag)
        np.testing.assert_array_equal(restored.calibration, self.model.calibration)
        np.testing.assert_array_equal(restored._mean, self.model._mean)
        np.testing.assert_array_equal(restored._std, self.model._std)
        self.assertEqual(list(restored._G.edges), list(self.model._G.edges))
        for name, tensor in self.model.ebm.state_dict().items():
            torch.testing.assert_close(restored.ebm.state_dict()[name], tensor)

    def test_base_loader_keeps_saved_subclass_and_its_attributes(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "tagged.pt"
            self.model.save(filename)
            restored = DEBTA.load(filename)
        self.assertIs(type(restored), TaggedDEBTA)
        self.assertEqual(restored.tag, "trained")
        np.testing.assert_array_equal(restored.calibration, self.model.calibration)

    def test_legacy_base_checkpoint_can_be_loaded_or_promoted_without_refitting(self):
        base = DEBTA(num_y=2, num_h=2, log_mlflow=False)
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "old_base.pt"
            base.save(filename)
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
            checkpoint.pop("extra_state")
            torch.save(checkpoint, filename)
            restored = DEBTA.load(filename)
            promoted = TaggedDEBTA.load(filename)
        self.assertIs(type(restored), DEBTA)
        self.assertIs(type(promoted), TaggedDEBTA)
        self.assertEqual(promoted.tag, "unfitted")
        self.assertIsNone(promoted.calibration)
        for name, tensor in base.ebm.state_dict().items():
            torch.testing.assert_close(promoted.ebm.state_dict()[name], tensor)

    def test_sibling_subclass_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "tagged.pt"
            self.model.save(filename)
            with self.assertRaisesRegex(TypeError, "incompatible"):
                OtherDEBTA.load(filename)

    def test_only_declared_attributes_are_saved(self):
        self.model.temporary_cache = lambda: None
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "tagged.pt"
            self.model.save(filename)
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        self.assertEqual(set(checkpoint["extra_state"]), {"tag", "calibration"})


if __name__ == "__main__":
    unittest.main()
