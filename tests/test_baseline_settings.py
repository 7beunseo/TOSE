from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "baselines" / "settings.py"
SPEC = importlib.util.spec_from_file_location("baseline_settings", SETTINGS_PATH)
SETTINGS_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SETTINGS_MODULE)


class BaselineSettingsTests(unittest.TestCase):
    def test_appendix_optimization_settings(self):
        settings = SETTINGS_MODULE.BASELINE_SETTINGS
        expected = {
            "remasker": ("AdamW", 1e-3, 0.01, 64),
            "pmae": ("AdamW", 1e-3, 0.01, 64),
            "cacti": ("AdamW", 1e-3, 1e-3, 128),
            "gain": ("Adam", 1e-3, 0.0, 128),
            "miwae": ("Adam", 1e-3, 0.0, 64),
            "dsan": ("Adam", 3e-3, 1e-5, 128),
            "deepifsac": ("AdamW", 1e-4, 1e-2, 128),
        }
        for method, values in expected.items():
            actual = settings[method]
            self.assertEqual(
                (actual["optimizer"], actual["lr"], actual["weight_decay"], actual["batch_size"]),
                values,
            )

    def test_method_specific_settings(self):
        settings = SETTINGS_MODULE.BASELINE_SETTINGS
        self.assertEqual(settings["pmae"]["method"]["a"], 0.05)
        self.assertEqual(settings["pmae"]["method"]["b"], 0.5)
        self.assertEqual(settings["cacti"]["architecture"]["encoder_depth"], 10)
        self.assertEqual(settings["cacti"]["architecture"]["decoder_depth"], 4)
        self.assertEqual(settings["miwae"]["architecture"]["observation_model"], "student_t")
        self.assertEqual(settings["miwae"]["architecture"]["latent_dim"], "ceil(input_dim / 2)")
        self.assertEqual(settings["deepifsac"]["method"]["cutmix_continuous_rate"], 0.3)
        self.assertEqual(settings["mice"]["max_iterations"], 10)
        self.assertEqual(settings["missforest"]["trees"], 100)

    def test_runner_code_applies_nondefault_appendix_values(self):
        remasker = (ROOT / "baselines" / "remasker" / "run_remasker_native_protocol.py").read_text()
        pmae = (ROOT / "baselines" / "PMAE" / "pmae_same_protocol.py").read_text()
        miwae = (ROOT / "baselines" / "MIWAE" / "run_fair_protocol.py").read_text()
        deepifsac = (ROOT / "baselines" / "DeepIFSAC" / "run_deepifsac_official_ssl_protocol.py").read_text()
        self.assertIn("weight_decay=self.weight_decay", remasker)
        self.assertIn("weight_decay=self.weight_decay", pmae)
        self.assertIn("math.ceil(bundle.train_x_norm.shape[1] / 2)", miwae)
        self.assertIn('default="miwae"', miwae)
        self.assertIn('default="t"', miwae)
        self.assertIn("default=0.3", deepifsac)

    def test_official_source_locks_are_full_hashes(self):
        sources = json.loads((ROOT / "baselines" / "official_sources.json").read_text())
        self.assertEqual(set(sources), {"remasker", "pmae", "cacti", "gain", "miwae", "dsan", "deepifsac"})
        for metadata in sources.values():
            self.assertRegex(metadata["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(metadata["url"].startswith("https://github.com/"))


if __name__ == "__main__":
    unittest.main()
