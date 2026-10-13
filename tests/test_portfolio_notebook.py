"""Protect unexecuted refactor cells and separately labeled historical artifacts."""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PortfolioNotebookTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(
            (ROOT / 'ISLES_2022_First_Colab_Notebook.ipynb').read_text()
        )
        self.portfolio = json.loads(
            (ROOT / 'notebooks' / 'ISLES_2022_Portfolio.ipynb').read_text()
        )

    def test_refactored_code_has_no_stale_outputs_and_uses_explicit_modules(self):
        original_code = [cell for cell in self.original['cells'] if cell['cell_type'] == 'code']
        portfolio_code = [cell for cell in self.portfolio['cells'] if cell['cell_type'] == 'code']
        self.assertLess(len(portfolio_code), len(original_code))
        for cell in portfolio_code:
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
            tree = ast.parse(''.join(cell['source']))
            self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree)))
        source = '\n'.join(''.join(cell['source']) for cell in portfolio_code)
        self.assertIn('from isles2022.preprocessing import preprocess_patient', source)
        self.assertIn('RUN_TRAINING = False', source)
        self.assertIn('RUN_EVALUATION = False', source)

    def test_historical_training_image_is_preserved_as_markdown_not_output(self):
        self.assertEqual(self.portfolio['nbformat'], self.original['nbformat'])
        self.assertEqual(self.portfolio['nbformat_minor'], self.original['nbformat_minor'])
        self.assertNotIn('widgets', self.portfolio['metadata'])
        expected = next(output['data']['image/png'] for output in self.original['cells'][40]['outputs']
                        if 'image/png' in output.get('data', {}))
        artifacts = [cell['attachments']['historical-training-loss.png']['image/png']
                     for cell in self.portfolio['cells']
                     if 'historical-training-loss.png' in cell.get('attachments', {})]
        self.assertEqual(artifacts, [expected])
        markdown = '\n'.join(''.join(cell['source']) for cell in self.portfolio['cells']
                             if cell['cell_type'] == 'markdown')
        self.assertIn('Historical training plot', markdown)
        self.assertIn('historical-test-examples.png', markdown)
        self.assertIn('0.5377', markdown)
        self.assertIn('0.5818', markdown)


if __name__ == '__main__':
    unittest.main()
