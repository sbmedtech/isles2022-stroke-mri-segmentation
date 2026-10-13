"""Execute the notebook's real selection helpers and synthetic Step 16 runs."""
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / 'ISLES_2022_First_Colab_Notebook.ipynb'
SOURCE = ''.join(json.loads(NOTEBOOK.read_text())['cells'][45]['source'])
TREE = ast.parse(SOURCE)
HELPERS = ast.Module(body=[node for node in TREE.body if isinstance(node, ast.FunctionDef)
                          and node.name in {'state_digest', 'context_digest', 'validate_selection',
                                            'establish_selection', 'save_atomic'}], type_ignores=[])
NS = dict(torch=torch, hashlib=hashlib, json=json, math=math, uuid=uuid)
exec(compile(HELPERS, str(NOTEBOOK), 'exec'), NS)


def legacy(epoch=23, weight=1.0, threshold=0.3):
    return {'epoch': epoch, 'model_state_dict': {'weight': torch.tensor([weight])},
            'optimizer_state_dict': {}, 'training_history': [],
            'validation_threshold': threshold}


def measure(candidate, threshold):
    return {'mean_dice': float(candidate['model_state_dict']['weight'][0]) / 10,
            'median_dice': 0.1, 'missed_cases': 0, 'false_positive_negative_cases': 0}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.context = {'splits': {'validation': ['patient-a']}, 'cache': 'sha256'}

    def start(self):
        return NS['establish_selection'](legacy(), self.context, measure)

    def test_new_baseline_uses_actual_weights_and_ignores_stale_best(self):
        latest, best, metrics = NS['establish_selection'](
            legacy(), self.context, measure, legacy(weight=9.0))
        self.assertAlmostEqual(metrics['mean_dice'], 0.1)
        self.assertEqual(best['epoch'], 23)
        self.assertEqual(best['selection']['threshold'], 0.3)
        self.assertEqual(latest['selection']['history'][0]['mean_dice'], metrics['mean_dice'])
        self.assertTrue(torch.equal(best['model_state_dict']['weight'], torch.tensor([1.])))

    def test_resume_remeasures_best_instead_of_trusting_history_scores(self):
        latest, best, _ = self.start()
        latest = copy.deepcopy(latest)
        latest['epoch'] = 28
        latest['selection']['history'][0]['mean_dice'] = 0.99
        _, actual_best, metrics = NS['establish_selection'](latest, self.context, measure, best)
        self.assertAlmostEqual(metrics['mean_dice'], 0.1)
        self.assertAlmostEqual(actual_best['selection']['history'][0]['mean_dice'], 0.1)

    def test_reject_missing_best_and_foreign_run(self):
        latest, best, _ = self.start()
        with self.assertRaises(ValueError):
            NS['establish_selection'](latest, self.context, measure)
        foreign = copy.deepcopy(best)
        foreign['selection']['run_id'] = 'another-run'
        with self.assertRaises(ValueError):
            NS['establish_selection'](latest, self.context, measure, foreign)

    def test_reject_changed_split_cache_config_threshold_or_weights(self):
        latest, best, _ = self.start()
        for context in [{'splits': {}}, {**self.context, 'cache': 'changed'},
                        {**self.context, 'model': 'changed'}]:
            with self.subTest(context=context), self.assertRaises(ValueError):
                NS['establish_selection'](latest, context, measure, best)
        foreign = copy.deepcopy(best)
        foreign['selection']['threshold'] = 0.25
        with self.assertRaises(ValueError):
            NS['establish_selection'](latest, self.context, measure, foreign)
        changed = copy.deepcopy(latest)
        changed['model_state_dict']['weight'] += 1
        with self.assertRaises(ValueError):
            NS['establish_selection'](changed, self.context, measure, best)

    def test_reject_unproven_legacy_resume_and_nonfinite_metric(self):
        with self.assertRaises(ValueError):
            NS['establish_selection'](legacy(epoch=43), self.context, measure)
        with self.assertRaises(ValueError):
            NS['establish_selection'](legacy(), self.context,
                                      lambda *_: {'mean_dice': float('nan')})

    def test_test_evaluation_rejects_foreign_run(self):
        _, best, _ = self.start()
        with self.assertRaises(ValueError):
            NS['validate_selection'](best, self.context, 'another-run')

    def test_step14_binds_threshold_to_evaluated_epoch23_weights(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = ''.join(notebook['cells'][43]['source'])
        persistence = source[source.index('# Bind the selected threshold'):]
        with TemporaryDirectory() as root:
            project = Path(root)
            (project / 'models').mkdir()
            model = torch.nn.Linear(1, 1)
            checkpoint = {'epoch': 23, 'model_state_dict': model.state_dict()}
            path = project / 'models' / 'checkpoint_latest.pt'
            torch.save(checkpoint, path)
            env = dict(torch=torch, PROJECT_DIR=project, model=model, best_threshold=0.3)
            exec(persistence, env)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['validation_threshold'], 0.3)
            with torch.no_grad():
                model.weight.add_(1)
            with self.assertRaises(ValueError):
                exec(persistence, env)
            self.assertEqual(torch.load(path, weights_only=True)['validation_threshold'], 0.3)

    def test_step16_overwrites_stale_baseline_and_survives_resume(self):
        # Run actual notebook control flow with a tiny model/loader and mocked
        # full-volume validation; no MONAI inference or patient data required.
        with TemporaryDirectory() as root:
            project = Path(root)
            models = project / 'models'; models.mkdir()
            cache = project / 'cache'; cache.mkdir()
            results = project / 'results'; results.mkdir()
            (cache / 'a.pt').write_bytes(b'train-cache')
            (cache / 'b.pt').write_bytes(b'validation-cache')
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            checkpoint = {'epoch': 23, 'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'training_history': [], 'validation_threshold': 0.3}
            torch.save(checkpoint, models / 'checkpoint_latest.pt')
            stale = {**checkpoint, 'epoch': 99}
            torch.save(stale, models / 'checkpoint_best_validation.pt')
            torch.save(stale, models / 'checkpoint_epoch23.pt')
            (results / 'extended_validation_history.csv').write_text('epoch,mean_dice\n99,0.99\n')
            env = dict(PROJECT_DIR=project, LOCAL_CACHE_DIR=cache,
                       model=model, optimizer=optimizer, device=torch.device('cpu'),
                       train_ids=['a'], val_ids=['b'], test_ids=['c'], PATCH_SIZE=(64,64,32),
                       extended_train_dataset=SimpleNamespace(patches_per_patient=4),
                       extended_train_loader=[{'image': torch.ones(1,1), 'label': torch.zeros(1,1)}],
                       loss_function=torch.nn.MSELoss(), preprocess_patient=lambda: None,
                       normalize_mri=lambda: None, extract_patch=lambda: None,
                       synthetic_evaluate=lambda *_args, **_kw: {
                           'mean_dice': 0.2, 'median_dice': 0.2, 'missed_cases': 0,
                           'false_positive_negative_cases': 0})
            # The actual loader normally exposes batch_size.
            class Loader(list):
                batch_size = 2
            env['extended_train_loader'] = Loader(env['extended_train_loader'])
            tree = copy.deepcopy(TREE)
            for index, node in enumerate(tree.body):
                if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_validation':
                    tree.body[index] = ast.parse('evaluate_validation = synthetic_evaluate').body[0]
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'END_EPOCH' for t in node.targets):
                    node.value = ast.Constant(28)
            tree = ast.fix_missing_locations(tree)
            with patch.object(inspect, 'getsource', return_value='unchanged source'):
                exec(compile(tree, str(NOTEBOOK), 'exec'), env)
                best = torch.load(models / 'checkpoint_best_validation.pt', weights_only=True)
                latest = torch.load(models / 'checkpoint_latest.pt', weights_only=True)
                self.assertEqual(best['epoch'], 23)
                self.assertEqual(latest['epoch'], 28)
                self.assertEqual(latest['selection']['threshold'], 0.3)
                self.assertEqual([r['epoch'] for r in latest['selection']['history']], [23, 28])
                self.assertEqual(best['selection']['run_id'], latest['selection']['run_id'])
                # No additional epochs; resume must retain the measured epoch-23 best.
                exec(compile(tree, str(NOTEBOOK), 'exec'), env)
                self.assertEqual(env['best_epoch'], 23)
                self.assertEqual(env['VALIDATION_THRESHOLD'], 0.3)
                # A genuinely improved validation score must save the same
                # epoch/weights as the latest checkpoint and its measured history.
                scores = iter([0.2, 0.4])
                env['synthetic_evaluate'] = lambda *_args, **_kw: {
                    'mean_dice': next(scores), 'median_dice': 0.2,
                    'missed_cases': 0, 'false_positive_negative_cases': 0}
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'END_EPOCH' for t in node.targets
                    ):
                        node.value = ast.Constant(33)
                exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), 'exec'), env)
                best = torch.load(models / 'checkpoint_best_validation.pt', weights_only=True)
                latest = torch.load(models / 'checkpoint_latest.pt', weights_only=True)
                self.assertEqual(best['epoch'], 33)
                self.assertEqual(latest['epoch'], 33)
                self.assertEqual(best['selection']['weights_hash'], latest['selection']['weights_hash'])
                self.assertEqual(best['selection']['history'][-1]['mean_dice'], 0.4)


if __name__ == '__main__':
    unittest.main()
