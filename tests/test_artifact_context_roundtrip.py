"""R1: full cell-45 context with real MONAI/AdamW objects and JSON artifacts."""
import ast
import copy
import hashlib
import inspect
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from monai.losses import DiceCELoss
from monai.networks.nets import UNet
import torch

import test_notebook_artifacts as artifact_tests


class AdamWContextRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.fixture = artifact_tests.NotebookArtifactTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.env = self.fixture.env
        # Actual notebook architecture/loss/optimizer, with nonempty AdamW state.
        model = UNet(spatial_dims=3, in_channels=3, out_channels=1,
                     channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2)
        loss = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        sample = torch.randn(1, 3, 16, 16, 8)
        label = torch.zeros(1, 1, 16, 16, 8)
        label[:, :, 4:8, 4:8, 2:4] = 1
        optimizer.zero_grad()
        loss(model(sample), label).backward()
        optimizer.step()
        self.assertTrue(optimizer.state)
        self.assertIsInstance(optimizer.param_groups[0]['betas'], tuple)
        model.load_state_dict = Mock(wraps=model.load_state_dict)
        ids = [f'sub-strokecase{i:04d}' for i in range(1, 251)]
        random.Random(42).shuffle(ids)
        train_ids, val_ids, test_ids = ids[:175], ids[175:212], ids[212:]
        self.assertTrue({'sub-strokecase0237', 'sub-strokecase0115'}.issubset(test_ids))
        self.fixture.patients = test_ids
        self.fixture.model = model
        self.env.update(model=model, optimizer=optimizer, loss_function=loss,
                        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                        cache_digest=hashlib.sha256(b'synthetic-cache'),
                        extract_patch=lambda *_: None, evaluate_validation=lambda *_: None,
                        PATCH_SIZE=(64, 64, 32), VALIDATE_EVERY=5,
                        extended_train_dataset=SimpleNamespace(patches_per_patient=4),
                        extended_train_loader=SimpleNamespace(batch_size=2))
        # Execute the actual full selection_context assignment, not a hand-built
        # dictionary that omits optimizer fields or simplifies loss/model settings.
        tree = ast.parse(''.join(artifact_tests.CELLS[45]['source']))
        node = next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'selection_context'
                            for target in node.targets))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(artifact_tests.NOTEBOOK), 'exec'), self.env)
        self.context = self.env['selection_context']
        self.assertEqual(self.context['optimizer'][0]['betas'], (0.9, 0.999))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_187_546)
        self.assertEqual([len(self.context['splits'][key]) for key in ['train', 'validation', 'test']],
                         [175, 37, 38])
        checkpoint = {
            'epoch': 43, 'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'selection': {'run_id': 'run-a', 'threshold': 0.3,
                          'context_hash': self.env['context_digest'](self.context),
                          'weights_hash': self.env['state_digest'](model.state_dict())},
        }
        self.fixture.checkpoint = checkpoint
        self.fixture.context = self.context
        self.env['checkpoint'] = checkpoint
        torch.save(checkpoint, self.fixture.checkpoint_path)
        # Substitute only costly inference. Cells 46–49, all serializers,
        # checkpoint validation, full 38-case CSV/PDF paths and metric checks run.
        self.fixture.inference = lambda **kwargs: torch.zeros_like(kwargs['inputs'][:, :1]).cpu()

    def test_complete_adamw_provenance_round_trips_csv_png_pdf(self):
        csv = self.fixture.generate_results()
        provenance = self.env['test_provenance']
        serialized = json.loads(json.dumps(provenance))
        self.assertEqual(serialized, provenance)
        self.assertEqual(self.env['context_digest'](self.context),
                         self.env['context_digest'](json.loads(json.dumps(self.context))))
        frame, manifest = self.env['read_test_artifact'](self.fixture.project, provenance)
        self.assertEqual(frame['patient_id'].tolist(), self.fixture.patients)
        self.assertEqual(len(frame), 38)
        for cell in [47, 48, 49]:
            self.fixture.run_cell(cell)
        for name in ['example_test_patients.png', 'infarct_size_vs_dice.png',
                     'all_test_patient_visualizations.pdf']:
            path = csv.parent / name
            exported = json.loads(path.with_suffix(path.suffix + '.manifest.json').read_text())
            self.assertEqual(exported['provenance'], provenance)
            self.assertEqual(exported['source_test_csv_sha256'], manifest['sha256'])
            self.assertEqual(exported['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        # Canonicalization must not mutate actual optimizer/training objects.
        self.assertIsInstance(self.env['optimizer'].param_groups[0]['betas'], tuple)
        self.assertIsInstance(self.context['optimizer'][0]['betas'], tuple)

    def test_full_context_consumers_reject_changed_identity_after_json_round_trip(self):
        csv = self.fixture.generate_results()
        path = csv.with_suffix('.csv.manifest.json')
        original = json.loads(path.read_text())
        cases = {
            'run_id': 'foreign-run', 'epoch': 28, 'weights_sha256': 'foreign-weights',
            'checkpoint_digest': 'foreign-checkpoint', 'threshold': 0.7,
            'patient_ids': list(reversed(self.fixture.patients)),
            'environment': {'python': 'foreign-environment'},
            'context': {**original['provenance']['context'], 'preprocessing_source': 'changed'},
        }
        for field, value in cases.items():
            changed = copy.deepcopy(original)
            changed['provenance'][field] = value
            path.write_text(json.dumps(changed))
            for cell in [47, 48, 49]:
                with self.subTest(field=field, cell=cell), self.assertRaises(ValueError):
                    self.fixture.run_cell(cell)
        for mutate in [
            lambda context: context['optimizer'][0].update(betas=[0.8, 0.999]),
            lambda context: context['splits']['train'].reverse(),
            lambda context: context['splits']['test'].reverse(),
            lambda context: context.update(cache_sha256='changed-cache'),
        ]:
            changed = copy.deepcopy(original)
            mutate(changed['provenance']['context'])
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.fixture.run_cell(48)
        path.write_text(json.dumps(original))
        csv.write_bytes(csv.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            self.fixture.run_cell(49)

    def test_raw_and_round_tripped_provenance_share_exact_paths_and_comparisons(self):
        csv = self.fixture.generate_results()
        raw = copy.deepcopy(self.env['test_provenance'])
        raw['context'] = self.context  # real AdamW tuple metadata, before JSON conversion
        loaded = json.loads(json.dumps(raw))
        self.assertEqual(self.env['test_artifact_directory'](self.fixture.project, raw),
                         self.env['test_artifact_directory'](self.fixture.project, loaded))
        for representation in [raw, loaded]:
            frame, manifest = self.env['read_test_artifact'](self.fixture.project, representation)
            self.assertEqual(len(frame), 38)
            figure = csv.parent / 'source-comparison.png'
            figure.write_bytes(b'synthetic figure serialization fixture')
            written = self.env['write_artifact_manifest'](figure, representation, manifest)
            self.assertEqual(written, json.loads(figure.with_suffix('.png.manifest.json').read_text()))
        # Preserve the pre-fix context/checkpoint digest encoding for compatibility.
        expected = hashlib.sha256(json.dumps(self.context, sort_keys=True).encode()).hexdigest()
        self.assertEqual(self.env['context_digest'](self.context), expected)


if __name__ == '__main__':
    unittest.main()
