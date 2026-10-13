"""Protect the portfolio companion's executable cells and historical record."""

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

    def test_ordered_code_cells_and_all_saved_outputs_are_identical(self):
        original_code = [cell for cell in self.original['cells'] if cell['cell_type'] == 'code']
        portfolio_code = [cell for cell in self.portfolio['cells'] if cell['cell_type'] == 'code']
        # Whole-cell equality covers source, comments, output image bytes,
        # execution counts, metadata and any cell IDs, not just syntax.
        self.assertEqual(portfolio_code, original_code)

    def test_notebook_format_runtime_and_widget_metadata_are_preserved(self):
        self.assertEqual(self.portfolio['nbformat'], self.original['nbformat'])
        self.assertEqual(self.portfolio['nbformat_minor'], self.original['nbformat_minor'])
        self.assertEqual(self.portfolio['metadata'], self.original['metadata'])


if __name__ == '__main__':
    unittest.main()
