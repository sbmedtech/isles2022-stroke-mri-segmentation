"""Execute real notebook cells 46–49 on synthetic volumes and disk artifacts."""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import monai.inferers
import numpy as np
import pandas as pd
import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / 'ISLES_2022_First_Colab_Notebook.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']
HELPER_NAMES = {
    'state_digest', 'context_digest', 'validate_selection', 'test_artifact_provenance',
    'test_artifact_directory', 'write_artifact_manifest', 'read_test_artifact',
    'save_test_artifact', 'check_figure_metrics',
}
HELPERS = ast.Module(body=[node for node in ast.parse(''.join(CELLS[45]['source'])).body
                          if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES],
                     type_ignores=[])
SOURCE_MARKER = 'synthetic preprocessing source'


class NotebookArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(lambda: plt.close('all'))
        self.project = Path(self.temporary.name)
        (self.project / 'models').mkdir()
        self.model = torch.nn.Conv3d(3, 1, 1)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
        self.model.load_state_dict = Mock(wraps=self.model.load_state_dict)
        self.patients = ['sub-strokecase0237', 'sub-strokecase0115']
        self.context = {
            'splits': {'train': ['train'], 'validation': ['validation'], 'test': self.patients},
            'preprocessing_source': SOURCE_MARKER * 2,
            'cache_sha256': 'synthetic-cache-content',
        }
        self.env = dict(torch=torch, np=np, pd=pd, hashlib=hashlib, json=json,
                        math=math, uuid=uuid, inspect=inspect, io=io, os=os,
                        importlib=importlib, platform=platform, re=re,
                        PROJECT_DIR=self.project, model=self.model, device=torch.device('cpu'),
                        selection_context=self.context, selection_run_id='run-a',
                        test_ids=self.patients, preprocess_patient=self.preprocess,
                        normalize_mri=lambda volume: volume)
        exec(compile(HELPERS, str(NOTEBOOK), 'exec'), self.env)
        self.checkpoint = self.make_checkpoint('run-a', 0.3)
        self.checkpoint_path = self.project / 'models' / 'checkpoint_best_validation.pt'
        torch.save(self.checkpoint, self.checkpoint_path)
        self.env['checkpoint'] = self.checkpoint
        self.inference_calls = []
        self.source_patch = patch.object(inspect, 'getsource', return_value=SOURCE_MARKER)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def make_checkpoint(self, run_id, threshold, bias=0.0):
        state = copy.deepcopy(self.model.state_dict())
        state['bias'].fill_(bias)
        return {
            'epoch': 43, 'model_state_dict': state,
            'selection': {'run_id': run_id, 'threshold': threshold,
                          'context_hash': self.env['context_digest'](self.context),
                          'weights_hash': self.env['state_digest'](state)},
        }

    def preprocess(self, patient_id):
        if patient_id not in self.patients:
            raise ValueError('Unexpected patient')
        image = torch.ones(3, 2, 2, 2)
        label = torch.zeros(1, 2, 2, 2)
        label.flatten()[:self.patients.index(patient_id) + 1] = 1
        return image, label

    def inference(self, **kwargs):
        self.inference_calls.append(copy.deepcopy(kwargs['predictor'].state_dict()))
        return kwargs['predictor'](kwargs['inputs']).detach().cpu()

    def run_cell(self, number):
        # Only expensive inference is substituted. All checkpoint load/validation,
        # artifact resolution, CSV I/O, metric checks, and figure/PDF paths are real.
        with patch.object(monai.inferers, 'sliding_window_inference', side_effect=self.inference), \
                patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            exec(compile(''.join(CELLS[number]['source']), f'{NOTEBOOK}:cell{number}', 'exec'), self.env)

    def generate_results(self):
        self.run_cell(46)
        self.model.load_state_dict.reset_mock()
        self.inference_calls.clear()
        return self.env['test_results_path']

    def test_real_cell47_validates_newly_loaded_foreign_checkpoint_before_using_it(self):
        self.generate_results()
        # In-memory A is valid, but the file now contains B with different weights/threshold.
        foreign = self.make_checkpoint('run-b', 0.7, bias=1.0)
        torch.save(foreign, self.checkpoint_path)
        with self.assertRaises(ValueError):
            self.run_cell(47)
        self.model.load_state_dict.assert_not_called()
        self.assertEqual(self.inference_calls, [])

    def test_real_cell47_uses_exact_loaded_weights_and_threshold_not_previous_object(self):
        self.generate_results()
        self.env['checkpoint'] = self.make_checkpoint('foreign-memory', 0.7, bias=1.0)
        self.run_cell(47)
        self.assertEqual(self.env['THRESHOLD'], 0.3)
        self.model.load_state_dict.assert_called_once()
        for key, value in self.checkpoint['model_state_dict'].items():
            torch.testing.assert_close(self.model.load_state_dict.call_args.args[0][key], value)
            for inferred_state in self.inference_calls:
                torch.testing.assert_close(inferred_state[key], value)
        self.assertTrue(self.env['figure_path'].is_file())

    def test_export_and_all_real_downstream_paths_have_matching_provenance(self):
        historical = self.project / 'results' / 'test_epoch43.csv'
        historical.parent.mkdir()
        historical.write_bytes(b'untouched historical results')
        csv = self.generate_results()
        self.assertIn('run-a', str(csv))
        manifest = json.loads(csv.with_suffix('.csv.manifest.json').read_text())
        provenance = manifest['provenance']
        self.assertEqual(provenance['run_id'], 'run-a')
        self.assertEqual(provenance['threshold'], 0.3)
        self.assertEqual(provenance['patient_ids'], self.patients)
        self.assertEqual(provenance['context'], self.context)
        self.assertEqual(provenance['weights_sha256'], self.env['state_digest'](self.checkpoint['model_state_dict']))
        self.assertEqual(len(provenance['checkpoint_digest']), 64)
        self.assertIn('python', provenance['environment'])
        for cell in [47, 48, 49]:
            self.run_cell(cell)
        directory = csv.parent
        for name in ['example_test_patients.png', 'infarct_size_vs_dice.png',
                     'all_test_patient_visualizations.pdf']:
            path = directory / name
            derived = json.loads(path.with_suffix(path.suffix + '.manifest.json').read_text())
            self.assertEqual(derived['provenance'], provenance)
            self.assertEqual(derived['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(derived['source_test_csv_sha256'], manifest['sha256'])
        self.assertEqual(historical.read_bytes(), b'untouched historical results')

    def test_real_downstream_paths_reject_cross_run_csv_even_at_same_epoch(self):
        csv_a = self.generate_results()
        cp_b = self.make_checkpoint('run-b', 0.3)
        provenance_b = self.env['test_artifact_provenance'](cp_b, self.context, 'run-b', self.patients)
        csv_b = self.env['save_test_artifact'](self.env['test_df'], self.project, provenance_b)
        # Deliberately swap both CSV and manifest into A's expected directory.
        shutil.copyfile(csv_b, csv_a)
        shutil.copyfile(csv_b.with_suffix('.csv.manifest.json'), csv_a.with_suffix('.csv.manifest.json'))
        for cell in [47, 48, 49]:
            with self.subTest(cell=cell), self.assertRaises(ValueError):
                self.run_cell(cell)
        self.model.load_state_dict.assert_not_called()
        self.assertEqual(self.inference_calls, [])
        self.assertFalse((csv_a.parent / 'all_test_patient_visualizations.pdf').exists())

    def test_real_scatter_path_rejects_each_mismatched_manifest_identity(self):
        csv = self.generate_results()
        path = csv.with_suffix('.csv.manifest.json')
        original = json.loads(path.read_text())
        for field, value in [('run_id', 'run-b'), ('checkpoint_digest', 'foreign-digest'),
                             ('weights_sha256', 'foreign-weights'), ('threshold', 0.7),
                             ('patient_ids', list(reversed(self.patients))),
                             ('context', {**self.context, 'preprocessing_source': 'foreign-source'})]:
            changed = copy.deepcopy(original)
            changed['provenance'][field] = value
            path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_cell(48)
        path.write_text(json.dumps(original))
        csv.write_bytes(csv.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            self.run_cell(48)

    def test_real_pdf_path_rejects_wrong_patient_order_and_metric_prediction_mismatch(self):
        csv = self.generate_results()
        frame = self.env['test_df'].copy()
        # Even a CSV with a fresh file checksum must preserve the split order.
        frame.iloc[::-1].to_csv(csv, index=False)
        self.env['write_artifact_manifest'](csv, self.env['test_provenance'])
        with self.assertRaises(ValueError):
            self.run_cell(49)
        # A matching manifest must also not pair differing predictions with its metrics.
        frame.loc[0, 'dice'] = 0.99
        self.env['save_test_artifact'](frame, self.project, self.env['test_provenance'])
        with self.assertRaises(ValueError):
            self.run_cell(49)
        self.assertFalse((csv.parent / 'all_test_patient_visualizations.pdf').exists())
        self.assertFalse((csv.parent / 'all_test_patient_visualizations.pdf.manifest.json').exists())

    def test_real_paths_reject_stale_weights_threshold_and_current_preprocessing(self):
        self.generate_results()
        for checkpoint in [self.make_checkpoint('run-a', 0.7),
                           self.make_checkpoint('run-a', 0.3, bias=1.0)]:
            torch.save(checkpoint, self.checkpoint_path)
            for cell in [47, 48, 49]:
                with self.subTest(cell=cell), self.assertRaises(ValueError):
                    self.run_cell(cell)
        torch.save(self.checkpoint, self.checkpoint_path)
        with patch.object(inspect, 'getsource', return_value='changed preprocessing'):
            for cell in [46, 47, 48, 49]:
                with self.subTest(cell=cell), self.assertRaises(ValueError):
                    self.run_cell(cell)

    def test_real_paths_do_not_adopt_legacy_csv_without_manifest(self):
        csv = self.generate_results()
        csv.with_suffix('.csv.manifest.json').unlink()
        for cell in [47, 48, 49]:
            with self.subTest(cell=cell), self.assertRaises(ValueError):
                self.run_cell(cell)


if __name__ == '__main__':
    unittest.main()
