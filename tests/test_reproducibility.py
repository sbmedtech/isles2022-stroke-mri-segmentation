import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from isles2022.reproducibility import environment_report, seed_everything, seed_worker


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.strict = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.benchmark = torch.backends.cudnn.benchmark
        self.cudnn = torch.backends.cudnn.deterministic

    def tearDown(self):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        torch.use_deterministic_algorithms(self.strict, warn_only=self.warn)
        torch.backends.cudnn.benchmark = self.benchmark
        torch.backends.cudnn.deterministic = self.cudnn

    def draw(self):
        generator = seed_everything(123, deterministic=True)
        return random.random(), np.random.rand(), torch.rand(4), torch.randperm(8, generator=generator)

    def test_repeatable_initial_draws_and_loader_order(self):
        a, b = self.draw(), self.draw()
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertTrue(torch.equal(a[3], b[3]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_worker_seed_uses_torch_seed(self):
        with patch("torch.initial_seed", return_value=2**32 + 7):
            seed_worker(0)
            actual = random.random(), np.random.rand()
        self.assertEqual(actual[0], random.Random(7).random())
        self.assertEqual(actual[1], np.random.RandomState(7).rand())

    def test_invalid_seed_and_cuda_configuration(self):
        for value in [-1, 2**32, True, 1.5]:
            with self.assertRaises(ValueError):
                seed_everything(value)
        with patch("torch.cuda.is_available", return_value=True), patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                seed_everything(deterministic=True)

    def test_report_is_serializable_and_does_not_dump_secrets(self):
        with patch.dict("os.environ", {"SYNTHETIC_SECRET": "do-not-report"}):
            report = environment_report()
        encoded = json.dumps(report)
        self.assertNotIn("do-not-report", encoded)
        self.assertEqual(report["packages"]["numpy"], np.__version__)
        self.assertIn("python", report)
        self.assertIsInstance(report["cuda_available"], bool)
