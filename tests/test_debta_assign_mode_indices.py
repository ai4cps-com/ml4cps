"""Known/unseen mode alignment and configurable sensor-distance weighting."""

import unittest

import numpy as np

from ml4cps.debta import DEBTA


class AssignModeIndicesTests(unittest.TestCase):
    def test_exact_matches_ignore_sensor_preference(self):
        modes, unknown = DEBTA.assign_mode_indices([0, 3], [[0], [10]], [3, 0],
                                         [[10000], [-10000]], [0], [1],
                                         sensor_distance_weight=1000000.)
        np.testing.assert_array_equal(modes, [1, 0])
        np.testing.assert_array_equal(unknown, [False, False])

    def test_mixed_codes_keep_the_original_unknown_flags(self):
        modes, unknown = DEBTA.assign_mode_indices([0, 1, 3], [[0], [10], [10]], [0, 3],
                                         [[0], [10]], [0], [1])
        np.testing.assert_array_equal(modes, [0, 1, 1])
        np.testing.assert_array_equal(unknown, [False, True, False])

    def test_weight_changes_the_unknown_mode_selection(self):
        args = ([3], [[0]], [0, 1], [[0], [10]], [0], [1])
        default, _ = DEBTA.assign_mode_indices(*args)
        changed, _ = DEBTA.assign_mode_indices(*args, sensor_distance_weight=.02)
        np.testing.assert_array_equal(default, [1])
        np.testing.assert_array_equal(changed, [0])

    def test_zero_weight_disables_sensor_distance(self):
        args = ([1], [[10]], [0, 3], [[0], [10]], [0], [1])
        default, _ = DEBTA.assign_mode_indices(*args)
        hamming_only, _ = DEBTA.assign_mode_indices(*args, sensor_distance_weight=0)
        np.testing.assert_array_equal(default, [1])
        np.testing.assert_array_equal(hamming_only, [0])
        with np.errstate(all='raise'):
            modes, flags = DEBTA.assign_mode_indices([1], [[np.nan]], [0, 3], [[np.nan], [np.nan]],
                                           None, [0], sensor_distance_weight=0)
        np.testing.assert_array_equal(modes, [0])
        np.testing.assert_array_equal(flags, [True])

    def test_sensor_distance_is_scaled_per_feature(self):
        args = ([1], [[0, 0]], [0, 3], [[2, 0], [0, 10]], [0, 0])
        scaled, _ = DEBTA.assign_mode_indices(*args, [1, 10])
        unscaled, _ = DEBTA.assign_mode_indices(*args, [1, 1])
        np.testing.assert_array_equal(scaled, [1])
        np.testing.assert_array_equal(unscaled, [0])

    def test_rejects_invalid_weights_even_for_known_codes(self):
        args = ([0], [[0]], [0, 1], [[0], [1]], [0], [1])
        for weight in (-1, np.nan, np.inf, -np.inf, None, '1e-6', [1e-6], 1j, True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                DEBTA.assign_mode_indices(*args, sensor_distance_weight=weight)

    def test_empty_input_retains_integer_index_and_boolean_flag_types(self):
        modes, unknown = DEBTA.assign_mode_indices([], np.empty((0, 1)), [0, 1], [[0], [1]], [0], [1])
        self.assertEqual(modes.shape, (0,))
        self.assertTrue(np.issubdtype(modes.dtype, np.integer))
        self.assertEqual(unknown.dtype, np.dtype(bool))
        self.assertEqual(unknown.shape, (0,))

    def test_instance_and_class_calls_share_the_same_method(self):
        model = DEBTA(num_y=1, num_h=2, log_mlflow=False)
        args = ([3], [[0]], [0, 1], [[0], [10]], [0], [1])
        class_result = DEBTA.assign_mode_indices(*args, sensor_distance_weight=.02)
        instance_result = model.assign_mode_indices(*args, sensor_distance_weight=.02)
        for actual, expected in zip(instance_result, class_result):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
