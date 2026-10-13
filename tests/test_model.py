"""Synthetic architecture and legacy checkpoint compatibility checks."""

import ast
import io
import json
from pathlib import Path
import unittest

import torch
from monai.networks.nets import UNet

from isles2022.config import ModelConfig
from isles2022.model import build_model


class ModelTests(unittest.TestCase):
    def test_config_matches_original_notebook_constructor(self):
        notebook = Path(__file__).resolve().parents[1] / "ISLES_2022_First_Colab_Notebook.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        tree = ast.parse("".join(cells[27]["source"]))
        call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "UNet"
        )
        arguments = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
        self.assertEqual(vars(ModelConfig()), arguments)

    def test_legacy_checkpoint_load_and_forward_parity(self):
        # Independent constructor copied from the notebook, with no wrapper.
        torch.manual_seed(42)
        original = UNet(
            spatial_dims=3, in_channels=3, out_channels=1,
            channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
        )
        torch.manual_seed(42)
        extracted = build_model()
        self.assertIs(type(extracted), UNet)
        self.assertEqual(sum(p.numel() for p in extracted.parameters()), 1_187_546)
        self.assertEqual(list(original.state_dict()), list(extracted.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, extracted.state_dict()[key], rtol=0, atol=0)

        # The notebook saves a dictionary, not a serialized model instance.
        optimizer = torch.optim.AdamW(original.parameters(), lr=1e-4, weight_decay=1e-5)
        checkpoint = {
            "epoch": 43,
            "model_state_dict": original.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": [{"epoch": 43, "train_loss": 0.6299}],
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=True)
        extracted.load_state_dict(loaded["model_state_dict"], strict=True)
        original.eval()
        extracted.eval()
        image = torch.randn(1, 3, 64, 64, 32)
        with torch.no_grad():
            expected = original(image)
            actual = extracted(image)
        self.assertEqual(actual.shape, (1, 1, 64, 64, 32))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
