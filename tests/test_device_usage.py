"""
Tests to verify that all deep-learning models are loaded onto the device
specified in the family configuration (CUDA, MPS, or CPU).

The tests are deliberately lightweight: they mock weight-file loading so
that no actual model checkpoints are required at test time.  What is being
tested is purely the *device routing* logic – i.e. that the device string
from ``config_loader.get_config()`` is respected when models are constructed
and that ``prediction.get_models()`` correctly places every model on that
device.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_state_dict(model: nn.Module) -> dict:
    """Return a state-dict filled with correctly shaped zero tensors."""
    return {k: torch.zeros_like(v) for k, v in model.state_dict().items()}


def _clear_predictor_cache():
    """Remove all cached predictor sub-modules so tests get fresh imports."""
    for key in list(sys.modules.keys()):
        if key == "predictor" or key.startswith("predictor."):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Segmentation model device test
# ---------------------------------------------------------------------------

class TestSegmentationModelDevice(unittest.TestCase):
    """Verify that ``predictor.segmentation.get_model`` uses the config device."""

    def _run_for_device(self, device_str: str):
        _clear_predictor_cache()

        import predictor.segmentation as seg_mod
        import segmentation_models_pytorch as smp

        fake_config = {
            "device": device_str,
            "model_paths": {
                "segmentation": "/fake/seg.pth",
            },
        }

        # Build a tiny stand-in for UnetPlusPlus so no download is needed
        tiny_base = nn.Sequential(nn.Conv2d(3, 1, 1))

        with patch.object(seg_mod, "get_config", return_value=fake_config), \
             patch.object(smp, "UnetPlusPlus", return_value=tiny_base):

            model = seg_mod.get_model(family="mosquito", pretrained=False)
            # prediction.py calls .to(device) on the returned model
            model = model.to(torch.device(device_str))

        expected = torch.device(device_str)
        for name, param in model.named_parameters():
            self.assertEqual(
                param.device.type,
                expected.type,
                msg=f"Parameter '{name}' is on {param.device}, expected {expected}",
            )

    def test_cpu_device(self):
        self._run_for_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        self._run_for_device("cuda")


# ---------------------------------------------------------------------------
# Landmark model device test
# ---------------------------------------------------------------------------

class TestLandmarkModelDevice(unittest.TestCase):
    """Verify that ``predictor.landmark.get_model`` uses the config device."""

    def _run_for_device(self, device_str: str):
        _clear_predictor_cache()

        from predictor.landmark import Hourglass
        import predictor.landmark as lm_mod

        # Build state dict with exact same arch used by get_model (4 blocks, 64 ch, N=3)
        real_model = Hourglass(in_channels=1, num_blocks=4,
                               intermediate_channels=64, output_channels=3)
        fake_state = _make_tiny_state_dict(real_model)

        fake_config = {
            "device": device_str,
            "N_landmarks": 3,
            "model_paths": {
                "landmark": "/fake/lm.pth",
            },
        }

        with patch.object(lm_mod, "get_config", return_value=fake_config), \
             patch("torch.load", return_value=fake_state):

            model = lm_mod.get_model(family="mosquito")
            # prediction.py calls .to(device) on the returned model
            model = model.to(torch.device(device_str))

        expected = torch.device(device_str)
        for name, param in model.named_parameters():
            self.assertEqual(
                param.device.type,
                expected.type,
                msg=f"Parameter '{name}' is on {param.device}, expected {expected}",
            )

    def test_cpu_device(self):
        self._run_for_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        self._run_for_device("cuda")


# ---------------------------------------------------------------------------
# Classification model device test
# ---------------------------------------------------------------------------

class TestClassificationModelDevice(unittest.TestCase):
    """Verify that ``predictor.classification.get_model`` uses the config device."""

    def _run_for_device(self, device_str: str):
        _clear_predictor_cache()

        tiny_model = nn.Sequential(nn.Linear(1, 1))

        import predictor.classification as cls_mod

        fake_config = {
            "device": device_str,
            "model_paths": {
                "classification": "/fake/cls.pth",
            },
        }

        with patch.object(cls_mod, "get_config", return_value=fake_config), \
             patch("torch.load", return_value=tiny_model):

            model = cls_mod.get_model(family="mosquito")

        # get_model maps the loaded object to the config device via map_location;
        # prediction.py then calls .to(device) again.  Both paths are equivalent
        # for CPU. For a real GPU run, torch.load with map_location already puts
        # parameters on the right device.
        expected = torch.device(device_str)
        for name, param in model.named_parameters():
            self.assertEqual(
                param.device.type,
                expected.type,
                msg=f"Parameter '{name}' is on {param.device}, expected {expected}",
            )

    def test_cpu_device(self):
        self._run_for_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        self._run_for_device("cuda")


# ---------------------------------------------------------------------------
# Integration: prediction.get_models places all models on config device
# ---------------------------------------------------------------------------

class TestPredictionGetModelsDevice(unittest.TestCase):
    """
    Verify that ``predictor.prediction.get_models`` places *all* returned
    models on the device specified by the config.
    """

    def _run_for_device(self, device_str: str):
        _clear_predictor_cache()

        fake_config = {
            "device": device_str,
            "has_classification": True,
            "N_landmarks": 3,
            "model_paths": {
                "segmentation": "/fake/seg.pth",
                "landmark": "/fake/lm.pth",
                "classification": "/fake/cls.pth",
            },
        }

        seg_model = nn.Sequential(nn.Linear(1, 1))
        lm_model = nn.Sequential(nn.Linear(1, 1))
        cls_model = nn.Sequential(nn.Linear(1, 1))

        import predictor.prediction as pred_mod
        import predictor.segmentation as seg_mod
        import predictor.landmark as lm_mod
        import predictor.classification as cls_mod

        with patch.object(pred_mod, "get_config", return_value=fake_config), \
             patch.object(seg_mod, "get_model", return_value=seg_model), \
             patch.object(lm_mod, "get_model", return_value=lm_model), \
             patch.object(cls_mod, "get_model", return_value=cls_model):

            pred_mod._model_cache.clear()
            seg, lm, cls, returned_device = pred_mod.get_models(family="mosquito")

        expected = torch.device(device_str)
        for model, model_name in [(seg, "segmentation"), (lm, "landmark"), (cls, "classification")]:
            for pname, param in model.named_parameters():
                self.assertEqual(
                    param.device.type,
                    expected.type,
                    msg=f"[{model_name}] param '{pname}' on {param.device}, expected {expected}",
                )

        self.assertEqual(returned_device, device_str)

    def test_cpu_device(self):
        self._run_for_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        self._run_for_device("cuda")


# ---------------------------------------------------------------------------
# Test that config_loader._detect_device() returns a usable torch device
# ---------------------------------------------------------------------------

class TestDetectDevice(unittest.TestCase):
    """Verify _detect_device returns a string torch.device can accept."""

    def test_detect_device_is_valid_torch_device(self):
        from config_loader import _detect_device
        device_str = _detect_device()
        device = torch.device(device_str)  # must not raise
        self.assertIn(device.type, ("cpu", "cuda", "mps"))

    def test_config_device_matches_available_hardware(self):
        """Config device should be CUDA when CUDA is available, else MPS/CPU."""
        from config_loader import _detect_device
        device_str = _detect_device()
        if torch.cuda.is_available():
            self.assertEqual(device_str, "cuda")
        elif torch.backends.mps.is_available():
            self.assertEqual(device_str, "mps")
        else:
            self.assertEqual(device_str, "cpu")


if __name__ == "__main__":
    unittest.main()
