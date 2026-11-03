import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_generation import normalize_diffputer_preprocessed as normalize
from data_generation import prepare_diffputer_dataset as prepare
from data_generation.strict_masking import sample_exact_split_masks


class FeatureLoadingTest(unittest.TestCase):
    def test_supervised_target_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "toy"
            info_dir = root / "Info"
            dataset_dir.mkdir()
            info_dir.mkdir()
            frame = pd.DataFrame(
                {
                    "number": [1.0, 2.0, 3.0, 4.0],
                    "category": ["a", "b", "a", "b"],
                    "label": [0, 1, 0, 1],
                }
            )
            frame.to_csv(dataset_dir / "data.csv", index=False)
            frame.iloc[:3].to_csv(dataset_dir / "train.csv", index=False)
            frame.iloc[3:].to_csv(dataset_dir / "test.csv", index=False)
            (info_dir / "toy.json").write_text(
                json.dumps(
                    {
                        "num_col_idx": [0],
                        "cat_col_idx": [1],
                        "target_col_idx": [2],
                    }
                ),
                encoding="utf-8",
            )

            old_data_dir, old_info_dir = prepare.DATA_DIR, prepare.INFO_DIR
            prepare.DATA_DIR, prepare.INFO_DIR = root, info_dir
            try:
                train_x, test_x = prepare.load_dataset("toy")
            finally:
                prepare.DATA_DIR, prepare.INFO_DIR = old_data_dir, old_info_dir

            self.assertEqual(train_x.shape, (3, 2))
            self.assertEqual(test_x.shape, (1, 2))
            np.testing.assert_array_equal(train_x[:, 0], np.array([1, 2, 3]))
            np.testing.assert_array_equal(train_x[:, 1], np.array([0, 1, 0]))

    def test_feature_only_mask_is_mapped_to_num_then_categorical_layout(self):
        feature_indices = [0, 2, 3]
        mask = np.array([[True, False, True]], dtype=bool)
        numeric = normalize.select_feature_mask(mask, feature_indices, [0, 3], 5)
        categorical = normalize.select_feature_mask(mask, feature_indices, [2], 5)
        np.testing.assert_array_equal(numeric, np.array([[True, True]]))
        np.testing.assert_array_equal(categorical, np.array([[False]]))


class ExactMaskingTest(unittest.TestCase):
    def test_each_partition_has_nearest_exact_overall_rate(self):
        train_prob = np.full((20, 4), 0.3, dtype=np.float64)
        test_prob = np.full((10, 4), 0.3, dtype=np.float64)
        result = sample_exact_split_masks(
            train_probabilities=train_prob,
            test_probabilities=test_prob,
            train_indices=np.arange(15),
            valid_indices=np.arange(15, 20),
            missing_columns=np.arange(4),
            overall_missing_rate=0.3,
            max_train_valid_gap=1.0,
            min_observed_rate=0.0,
            min_missing_rate=0.0,
            min_count=0,
            max_tries=5,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result.train_mask[:15].sum(), round(0.3 * 15 * 4))
        self.assertEqual(result.train_mask[15:].sum(), round(0.3 * 5 * 4))
        self.assertEqual(result.test_mask.sum(), round(0.3 * 10 * 4))

    def test_ineligible_mar_predictors_remain_observed(self):
        train_prob = np.zeros((12, 4), dtype=np.float64)
        test_prob = np.zeros((6, 4), dtype=np.float64)
        train_prob[:, 2:] = 0.6
        test_prob[:, 2:] = 0.6
        result = sample_exact_split_masks(
            train_probabilities=train_prob,
            test_probabilities=test_prob,
            train_indices=np.arange(8),
            valid_indices=np.arange(8, 12),
            missing_columns=np.array([2, 3]),
            overall_missing_rate=0.3,
            max_train_valid_gap=1.0,
            min_observed_rate=0.0,
            min_missing_rate=0.0,
            min_count=0,
            max_tries=5,
            rng=np.random.default_rng(11),
        )
        self.assertFalse(result.train_mask[:, :2].any())
        self.assertFalse(result.test_mask[:, :2].any())


if __name__ == "__main__":
    unittest.main()
