"""Direct reference parity and synthetic integration for the executable refactor."""

import ast
import contextlib
import copy
from functools import partial
import hashlib
import inspect
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from isles2022 import checkpoints, data, datasets, evaluation, preprocessing, provenance, training
from isles2022.config import ExperimentConfig, ProjectPaths
from isles2022.visualization import plot_test_examples, plot_lesion_size, write_test_pdf

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "ISLES_2022_First_Colab_Notebook.ipynb").read_text())


def reference_functions(cells, names, **extra):
    namespace = dict(np=np, torch=torch, F=F, nib=nib, resample_from_to=resample_from_to,
                     pd=pd, json=json, hashlib=hashlib, inspect=inspect,
                     sliding_window_inference=sliding_window_inference, PATCH_SIZE=(64, 64, 32))
    namespace.update(extra)
    nodes = [node for index in cells for node in ast.parse("".join(REFERENCE["cells"][index]["source"])).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<reference>", "exec"), namespace)
    return namespace


class PreprocessingParityTests(unittest.TestCase):
    def test_normalization_matches_reference_for_finite_nonfinite_and_empty_inputs(self):
        ref = reference_functions([21], {"normalize_mri"})["normalize_mri"]
        for volume in [np.zeros((2, 3, 4), dtype=np.float32), np.ones((2, 3, 4), dtype=np.float32),
                       np.array([0, -4, 2, 8, np.nan, np.inf, -np.inf], dtype=np.float32)]:
            np.testing.assert_array_equal(preprocessing.normalize_mri(volume), ref(volume))

    def write_case(self, root, adc_affine=None, mask_shape=(8, 9, 5)):
        files = preprocessing.patient_files("sub-strokecase0001", root)
        affine = np.diag([2.0, 2.5, 3.0, 1.0])
        volume = np.arange(8 * 9 * 5, dtype=np.float32).reshape(8, 9, 5)
        volumes = {"DWI": volume, "ADC": np.flip(volume, axis=0).copy(),
                   "FLAIR": np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3),
                   "Mask": np.zeros(mask_shape, dtype=np.float32)}
        volumes["Mask"][1, 2, 3] = 1
        for name, path in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            grid = affine.copy()
            if name == "FLAIR":
                grid = np.diag([4.0, 5.0, 6.0, 1.0])
            if name == "ADC" and adc_affine is not None:
                grid = adc_affine
            nib.save(nib.Nifti1Image(volumes[name], grid), path)
        return files

    def test_nifti_preprocessing_tensor_channel_and_resampling_parity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_case(root)
            ref = reference_functions([21, 22], {"normalize_mri", "preprocess_patient"}, DATASET_DIR=root)
            actual = preprocessing.preprocess_patient("sub-strokecase0001", root)
            expected = ref["preprocess_patient"]("sub-strokecase0001")
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assertEqual(actual[0].shape, (3, 8, 9, 5))
            self.assertEqual(int(actual[1].sum()), 1)
            self.assertEqual(actual[0].dtype, torch.float32)

    def test_geometry_mismatch_and_missing_label_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            changed = np.diag([2.0, 2.5, 3.0, 1.0]); changed[0, 3] = 4
            files = self.write_case(root, adc_affine=changed)
            with self.assertRaises(ValueError):
                preprocessing.preprocess_patient("sub-strokecase0001", root)
            self.write_case(root)
            files["Mask"].unlink()
            with self.assertRaises(FileNotFoundError):
                preprocessing.preprocess_patient("sub-strokecase0001", root)


class SplitAndPatchParityTests(unittest.TestCase):
    def test_partition_matches_complete_reference_cell_and_rejects_existing_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(1, 251):
                (root / "data" / f"sub-strokecase{index:04d}").mkdir(parents=True)
            tree = ast.parse("".join(REFERENCE["cells"][19]["source"]))
            excluded = {"PROJECT_DIR", "DATASET_DIR", "SPLITS_DIR"}
            tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id in excluded for target in node.targets))]
            (root / "splits").mkdir()
            namespace = dict(DATASET_DIR=root / "data", SPLITS_DIR=root / "splits")
            previous = random.getstate()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(tree, "<reference split>", "exec"), namespace)
            finally:
                random.setstate(previous)
            path = root / "splits" / "patient_splits.json"
            actual = data.load_or_create_splits(root / "data", path)
            self.assertEqual(actual, namespace["split_data"])
            changed = copy.deepcopy(actual); changed["train"].reverse()
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                data.load_or_create_splits(root / "data", path)
            with self.assertRaises(ValueError):
                data.patient_splits(["duplicate"] * 250)

    def test_patch_values_and_rng_state_match_positive_random_and_empty_cases(self):
        ref = reference_functions([24], {"extract_patch"})["extract_patch"]
        for shape in [(12, 13, 5), (70, 72, 40)]:
            image = torch.arange(3 * np.prod(shape), dtype=torch.float32).reshape(3, *shape)
            label = torch.zeros(1, *shape); label[0, 1, 2, 3] = 1
            for positive, expert in [(True, label), (False, label), (True, torch.zeros_like(label))]:
                with self.subTest(shape=shape, positive=positive):
                    torch.manual_seed(123)
                    state = torch.get_rng_state()
                    expected = ref(image, expert, positive)
                    after = torch.get_rng_state()
                    torch.set_rng_state(state)
                    actual = datasets.extract_patch(image, expert, positive)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), after))

    def test_cached_dataset_policy_matches_reference_and_loader_defaults(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            image = torch.randn(3, 12, 13, 5); label = torch.zeros(1, 12, 13, 5)
            label[0, 1, 2, 3] = 1
            torch.save({"image": image, "label": label}, cache / "a.pt")
            ref = reference_functions([24, 33], {"extract_patch", "CachedISLESDataset"},
                                      LOCAL_CACHE_DIR=cache, Dataset=torch.utils.data.Dataset)
            expected = ref["CachedISLESDataset"](["a"], patches_per_patient=4)
            actual = datasets.CachedPatchDataset(["a"], cache, patches_per_patient=4)
            for index in range(4):
                torch.manual_seed(90 + index); state = torch.get_rng_state()
                before = expected[index]
                torch.set_rng_state(state); after = actual[index]
                for key in ("image", "label"):
                    torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
            loader = datasets.training_loader(["a"], cache, 4, ExperimentConfig(), torch.device("cpu"))
            self.assertEqual((loader.batch_size, loader.num_workers, loader.pin_memory), (2, 0, False))
            self.assertIsInstance(loader.sampler, torch.utils.data.RandomSampler)


class TrainingParityTests(unittest.TestCase):
    def test_stage_continuation_restores_weights_optimizer_and_history(self):
        with TemporaryDirectory() as directory:
            paths = ProjectPaths(Path(directory), Path(directory) / "cache")
            model = torch.nn.Conv3d(3, 1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            loss = torch.nn.MSELoss()
            batches = [{"image": torch.ones(1, 3, 4, 5, 3), "label": torch.zeros(1, 1, 4, 5, 3)}]
            first = training.train_until(model, optimizer, loss, batches, "cpu", paths, 1)
            frozen_digest = checkpoints.state_digest(first["model_state_dict"])
            expected_model = copy.deepcopy(model)
            expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=1e-4, weight_decay=1e-5)
            frozen_optimizer = copy.deepcopy(first["optimizer_state_dict"])
            expected_optimizer.load_state_dict(copy.deepcopy(first["optimizer_state_dict"]))
            training.train_epoch(expected_model, expected_optimizer, loss, batches, "cpu")
            with torch.no_grad(): model.weight.add_(9)
            resumed = training.train_until(model, optimizer, loss, batches, "cpu", paths, 2, checkpoint=first)
            self.assertEqual([row["epoch"] for row in resumed["training_history"]], [1, 2])
            self.assertEqual(checkpoints.state_digest(first["model_state_dict"]), frozen_digest)
            for parameter, state in frozen_optimizer["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(first["optimizer_state_dict"]["state"][parameter][name],
                                               value, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_model.state_dict()[key], rtol=0, atol=0)
            saved = torch.load(paths.models / "checkpoint_latest.pt", weights_only=True)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(json.loads((paths.models / "training_history.json").read_text()),
                             resumed["training_history"])

    def test_component_defaults_and_diagnostic_update_are_explicit(self):
        model, loss, optimizer = training.training_components(torch.device("cpu"))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1_187_546)
        self.assertEqual((loss.lambda_dice, loss.lambda_ce, loss.dice.sigmoid, loss.dice.squared_pred),
                         (1.0, 1.0, True, False))
        self.assertEqual((optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"]),
                         (1e-4, 1e-5))
        tiny = torch.nn.Conv3d(3, 1, 1)
        opt = torch.optim.AdamW(tiny.parameters(), lr=1e-4, weight_decay=1e-5)
        before = tiny.weight.detach().clone()
        batch = {"image": torch.ones(1, 3, 4, 5, 3), "label": torch.zeros(1, 1, 4, 5, 3)}
        training.diagnostic_optimizer_step(tiny, opt, loss, batch, torch.device("cpu"))
        self.assertFalse(torch.equal(tiny.weight, before))

    def test_actual_reference_epoch_updates_and_batch_mean_match_extraction(self):
        with TemporaryDirectory() as directory:
            model = torch.nn.Conv3d(3, 1, 1)
            ref_model = copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            ref_optimizer = torch.optim.AdamW(ref_model.parameters(), lr=1e-4, weight_decay=1e-5)
            loss = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1, lambda_ce=1)
            batches = [{"image": torch.randn(2, 3, 4, 5, 3),
                        "label": (torch.rand(2, 1, 4, 5, 3) > 0.8).float()} for _ in range(2)]
            class QuietProgress(list):
                def set_postfix(self, **kwargs): pass
            namespace = dict(model=ref_model, optimizer=ref_optimizer, loss_function=loss,
                             cached_train_loader=batches, device=torch.device("cpu"), NUM_EPOCHS=1,
                             training_history=[], CHECKPOINT_DIR=Path(directory),
                             torch=torch, json=json, time=time, tqdm=lambda items, **kwargs: QuietProgress(items))
            loop = next(node for node in ast.parse("".join(REFERENCE["cells"][35]["source"])).body
                        if isinstance(node, ast.For))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=[loop], type_ignores=[]), "<reference epoch>", "exec"), namespace)
            actual_loss, _ = training.train_epoch(model, optimizer, loss, batches, torch.device("cpu"))
            self.assertEqual(actual_loss, namespace["training_history"][0]["train_loss"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, ref_model.state_dict()[key], rtol=0, atol=0)
            # Parameter objects differ; their serialized optimizer order is identical.
            for actual, expected in zip(optimizer.state_dict()["state"].values(), ref_optimizer.state_dict()["state"].values()):
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_checkpoint_raw_compatibility_snapshot_and_threshold_binding(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            model = torch.nn.Conv3d(3, 1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            saved = training.make_checkpoint(model, optimizer, [], 23)
            self.assertEqual(set(saved), {"epoch", "model_state_dict", "optimizer_state_dict", "training_history"})
            original_digest = checkpoints.state_digest(saved["model_state_dict"])
            with torch.no_grad(): model.weight.add_(1)
            self.assertEqual(checkpoints.state_digest(saved["model_state_dict"]), original_digest)
            checkpoints.save_atomic(saved, path)
            checkpoints.load_checkpoint(path, model, optimizer)
            bound = training.persist_validation_threshold(path, model, 0.3)
            self.assertEqual(bound["validation_threshold"], 0.3)
            with torch.no_grad(): model.weight.add_(1)
            with self.assertRaises(ValueError): training.persist_validation_threshold(path, model, 0.3)

    def test_measured_selection_cadence_and_improved_best_are_preserved(self):
        with TemporaryDirectory() as directory:
            paths = ProjectPaths(Path(directory), Path(directory) / "cache"); paths.create()
            model = torch.nn.Conv3d(3, 1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            checkpoint = training.make_checkpoint(model, optimizer, [], 23)
            checkpoint["validation_threshold"] = 0.3
            checkpoints.save_atomic(checkpoint, paths.models / "checkpoint_latest.pt")
            context = {"test": "synthetic context"}
            splits = {"train": ["a"], "validation": ["b"], "test": ["c"]}
            batches = [{"image": torch.ones(1, 3, 4, 5, 3), "label": torch.zeros(1, 1, 4, 5, 3)}]
            scores = iter([0.2, 0.4])
            def measured(*args):
                return {"mean_dice": next(scores), "median_dice": 0.2,
                        "missed_cases": 0, "false_positive_negative_cases": 0}
            with patch.object(training, "evaluate_validation", side_effect=measured) as evaluate:
                latest, history = training.train_with_validation(
                    model, optimizer, torch.nn.MSELoss(), batches, torch.device("cpu"),
                    paths, splits, context, ExperimentConfig(final_epoch=28))
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(history["epoch"].tolist(), [23, 28])
            best = torch.load(paths.models / "checkpoint_best_validation.pt", weights_only=True)
            self.assertEqual((best["epoch"], latest["epoch"], best["selection"]["threshold"]), (28, 28, 0.3))
            self.assertEqual(best["selection"]["weights_hash"], latest["selection"]["weights_hash"])

    def test_nonfinite_loss_is_rejected_before_weight_update(self):
        model = torch.nn.Conv3d(3, 1, 1); optimizer = torch.optim.AdamW(model.parameters())
        before = model.weight.detach().clone()
        batch = {"image": torch.ones(1, 3, 2, 2, 2), "label": torch.zeros(1, 1, 2, 2, 2)}
        with self.assertRaises(RuntimeError):
            training.train_epoch(model, optimizer, lambda x, y: x.mean() * float("nan"), [batch], "cpu")
        self.assertTrue(torch.equal(model.weight, before))

    def test_diagnostic_retains_reference_valid_gradient_check_before_step(self):
        model = torch.nn.Conv3d(3, 1, 1, bias=False)
        with torch.no_grad(): model.weight.zero_()
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.step = Mock()
        batch = {"image": torch.ones(1, 3, 2, 2, 2), "label": torch.zeros(1, 1, 2, 2, 2)}
        with self.assertRaises(AssertionError):
            training.diagnostic_optimizer_step(model, optimizer, lambda x, y: torch.sqrt(x).mean(), batch, "cpu")
        optimizer.step.assert_not_called()


class EvaluationAndArtifactParityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.addCleanup(lambda: plt.close("all"))
        self.paths = ProjectPaths(Path(self.temporary.name), Path(self.temporary.name) / "cache")
        self.paths.create()
        self.model = torch.nn.Conv3d(3, 1, 1)
        with torch.no_grad(): self.model.weight.zero_(); self.model.bias.zero_()
        self.ids = ["a", "b", "empty"]
        for index, patient_id in enumerate(self.ids):
            image = torch.ones(3, 8, 9, 5); label = torch.zeros(1, 8, 9, 5)
            if patient_id != "empty": label.flatten()[:index + 1] = 1
            torch.save({"image": image, "label": label}, self.paths.cache / f"{patient_id}.pt")
        self.read = partial(datasets.load_cached_patient, cache_dir=self.paths.cache)
        self.config = ExperimentConfig()

    def test_full_volume_validation_matches_actual_reference_function(self):
        ref = reference_functions([45], {"evaluate_validation"},
                                  LOCAL_CACHE_DIR=self.paths.cache, device=torch.device("cpu"))
        expected = ref["evaluate_validation"](self.model, self.ids, threshold=0.25)
        actual = evaluation.evaluate_validation(self.model, self.ids, self.read, 0.25, "cpu", self.config)
        self.assertEqual(actual, expected)

    def test_threshold_summary_matches_reference_cell_including_ties(self):
        checkpoint = {"epoch": 23, "model_state_dict": self.model.state_dict()}
        torch.save(checkpoint, self.paths.models / "checkpoint_latest.pt")
        namespace = dict(model=self.model, device=torch.device("cpu"), val_ids=self.ids,
                         LOCAL_CACHE_DIR=self.paths.cache, PROJECT_DIR=self.paths.root)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile("".join(REFERENCE["cells"][43]["source"]), "<reference sweep>", "exec"), namespace)
        threshold, details, summary = evaluation.select_threshold(self.model, self.ids, self.read, "cpu")
        self.assertEqual(threshold, namespace["best_threshold"])
        pd.testing.assert_frame_equal(summary, namespace["summary"])
        for key in namespace["threshold_df"].columns:
            pd.testing.assert_series_equal(details[key], namespace["threshold_df"][key])

    def context(self):
        return {"splits": {"train": [], "validation": [], "test": self.ids},
                "preprocessing_source": provenance.preprocessing_source(),
                "optimizer": [{"betas": (0.9, 0.999)}]}

    def selected(self, context, run_id="run-a"):
        return {"epoch": 43, "model_state_dict": copy.deepcopy(self.model.state_dict()),
                "selection": {"run_id": run_id, "threshold": 0.3,
                              "context_hash": checkpoints.context_digest(context),
                              "weights_hash": checkpoints.state_digest(self.model.state_dict())}}

    def test_module_csv_figures_pdf_roundtrip_and_cross_run_rejection(self):
        context = self.context(); selected = self.selected(context)
        identity = provenance.artifact_provenance(selected, context, "run-a", self.ids)
        rows = evaluation.evaluate_patients(self.model, self.ids, self.read, 0.3, "cpu")
        csv = provenance.save_test_artifact(rows, self.paths.root, identity)
        loaded, source = provenance.read_test_artifact(self.paths.root, json.loads(json.dumps(identity)))
        self.assertEqual(len(loaded), 3)
        plot_test_examples(self.model, selected, context, "run-a", self.ids[:2], self.read,
                           "cpu", self.config, self.paths.root)
        plot_lesion_size(selected, context, "run-a", self.paths.root)
        pdf = write_test_pdf(self.model, selected, context, "run-a", self.read,
                              "cpu", self.config, self.paths.root)
        manifest = json.loads(pdf.with_suffix(".pdf.manifest.json").read_text())
        self.assertEqual(manifest["source_test_csv_sha256"], source["sha256"])
        self.assertEqual(manifest["provenance"], identity)
        foreign = copy.deepcopy(identity); foreign["run_id"] = "run-b"
        path = csv.with_suffix(".csv.manifest.json")
        path.write_text(json.dumps({"provenance": foreign, "sha256": source["sha256"]}))
        with self.assertRaises(ValueError):
            write_test_pdf(self.model, selected, context, "run-a", self.read, "cpu", self.config, self.paths.root)

    def test_selected_loading_rejects_foreign_weights_before_model_loading(self):
        context = self.context(); selected = self.selected(context, "foreign")
        path = self.paths.models / "foreign.pt"; torch.save(selected, path)
        model = Mock()
        with self.assertRaises(ValueError):
            checkpoints.load_selected_checkpoint(path, model, context, "run-a")
        model.load_state_dict.assert_not_called()

    def test_selection_context_keys_cache_and_actual_adamw_values_match_reference(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        loss = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1, lambda_ce=1)
        splits = {"train": ["a"], "validation": ["b"], "test": ["empty"]}
        digest = hashlib.sha256()
        for patient_id in splits["train"] + splits["validation"]:
            digest.update(patient_id.encode()); digest.update((self.paths.cache / f"{patient_id}.pt").read_bytes())
        source = "".join(REFERENCE["cells"][45]["source"])
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "selection_context" for target in node.targets))
        ns = dict(train_ids=splits["train"], val_ids=splits["validation"], test_ids=splits["test"],
                  cache_digest=digest, inspect=inspect, preprocess_patient=preprocessing.preprocess_patient,
                  normalize_mri=preprocessing.normalize_mri, extract_patch=datasets.extract_patch,
                  model=self.model, optimizer=optimizer, loss_function=loss,
                  evaluate_validation=evaluation.evaluate_validation, PATCH_SIZE=(64, 64, 32),
                  extended_train_dataset=type("D", (), {"patches_per_patient": 4})(),
                  extended_train_loader=type("L", (), {"batch_size": 2})(), VALIDATE_EVERY=5, torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<reference context>", "exec"), ns)
        actual = provenance.selection_context(self.paths, splits, self.model, optimizer, loss, self.config)
        self.assertEqual(actual, ns["selection_context"])


class DownloadTests(unittest.TestCase):
    def test_complete_archive_reuse_only_fetches_metadata(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ISLES-2022.zip"; path.write_bytes(b"archive")
            response = Mock(); response.json.return_value = {
                "files": [{"key": path.name, "size": 7, "links": {"self": "https://zenodo.org/fixture"}}]}
            with patch("requests.get", return_value=response) as get:
                self.assertEqual(data.download_dataset(Path(directory)), path)
            self.assertEqual(get.call_count, 1)

    def test_streamed_download_and_incomplete_size_rejection(self):
        from unittest.mock import MagicMock
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = Mock(); metadata.json.return_value = {
                "files": [{"key": "ISLES-2022.zip", "size": 7,
                           "links": {"self": "https://zenodo.org/fixture"}}]}
            stream = Mock(); stream.iter_content.return_value = [b"arc", b"", b"hive"]
            response = MagicMock(); response.__enter__.return_value = stream
            with patch("requests.get", side_effect=[metadata, response]):
                path = data.download_dataset(root)
            self.assertEqual(path.read_bytes(), b"archive")
            stream.iter_content.assert_called_once_with(chunk_size=8 * 1024 * 1024)
            path.unlink()
            stream.iter_content.return_value = [b"short"]
            with patch("requests.get", side_effect=[metadata, response]), self.assertRaises(ValueError):
                data.download_dataset(root)

    def test_archive_extraction_returns_expected_dataset_root(self):
        import zipfile
        with TemporaryDirectory() as directory:
            root = Path(directory); archive = root / "ISLES-2022.zip"
            with zipfile.ZipFile(archive, "w") as contents:
                contents.writestr("ISLES-2022/sub-strokecase0001/placeholder", "fixture")
            result = data.extract_dataset(archive, root / "extracted")
            self.assertTrue((result / "sub-strokecase0001" / "placeholder").is_file())


if __name__ == "__main__":
    unittest.main()
