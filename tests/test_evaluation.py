"""Synthetic calculation tests only; no patient data or saved results."""

import unittest

import numpy as np

from evaluation import bootstrap_patient_ci, infarct_volume_ml, nifti_infarct_volume_ml, zero_overlap


class VolumeTests(unittest.TestCase):
    def test_anisotropic_volume_and_empty(self):
        affine = np.diag([2., 3., 5., 1.])
        mask = np.ones((2, 5, 10))
        self.assertAlmostEqual(infarct_volume_ml(mask, affine, spatial_unit="mm"), 3.)
        self.assertEqual(infarct_volume_ml(mask * 0, affine, spatial_unit="mm"), 0.)

    def test_shear_reflection_translation(self):
        affine = np.array([[-2, 1, 0, 100], [0, 3, 1, 50], [0, 0, 5, -10], [0, 0, 0, 1.]])
        self.assertAlmostEqual(infarct_volume_ml(np.ones((1, 1, 1)), affine, spatial_unit="mm"), .03)

    def test_unit_conversion(self):
        mask = np.ones((2, 5, 10))
        for unit, scale in [("mm", 1), ("meter", .001), ("micron", 1000)]:
            affine = np.diag([2 * scale, 3 * scale, 5 * scale, 1])
            self.assertAlmostEqual(infarct_volume_ml(mask, affine, spatial_unit=unit), 3.)

    def test_invalid_geometry_and_masks(self):
        mask = np.ones((1, 1, 1))
        for affine in [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan), np.diag([0, 1, 1, 1])]:
            with self.subTest(affine=affine), self.assertRaises(ValueError):
                infarct_volume_ml(mask, affine, spatial_unit="mm")
        for bad in [np.ones((2, 2)), np.empty((0, 2, 2)), mask * .5, mask * np.nan, mask * 2]:
            with self.subTest(mask=bad), self.assertRaises(ValueError):
                infarct_volume_ml(bad, np.eye(4), spatial_unit="mm")
        with self.assertRaises(ValueError):
            infarct_volume_ml(mask, np.eye(4), spatial_unit="unknown")

    def test_nifti_header_units(self):
        import nibabel as nib
        image = nib.Nifti1Image(np.ones((2, 5, 10)), np.diag([2., 3., 5., 1.]))
        with self.assertRaises(ValueError):
            nifti_infarct_volume_ml(image)
        image.header.set_xyzt_units("mm")
        self.assertAlmostEqual(nifti_infarct_volume_ml(image), 3.)


class OverlapTests(unittest.TestCase):
    def test_all_empty_and_disjoint_cases(self):
        empty = np.zeros((2, 2, 2), dtype=bool)
        expert = empty.copy(); expert[0, 0, 0] = True
        disjoint = empty.copy(); disjoint[1, 1, 1] = True
        self.assertTrue(zero_overlap(expert, empty))
        self.assertTrue(zero_overlap(expert, disjoint))
        self.assertFalse(zero_overlap(expert, expert))
        self.assertFalse(zero_overlap(empty, empty))
        self.assertFalse(zero_overlap(empty, disjoint))
        self.assertFalse(zero_overlap(expert, expert | disjoint))

    def test_reject_shape_and_probabilities(self):
        with self.assertRaises(ValueError):
            zero_overlap(np.zeros((2, 2, 2)), np.zeros((3, 2, 2)))
        with self.assertRaises(ValueError):
            zero_overlap(np.ones((2, 2, 2)), np.full((2, 2, 2), .25))


class BootstrapTests(unittest.TestCase):
    def test_known_two_patient_distribution(self):
        # For [0, 1], bootstrap means are 0, .5, 1 with probabilities .25, .5, .25.
        result = bootstrap_patient_ci(["a", "b"], [0, 1], n_resamples=10000)
        self.assertEqual((result.estimate, result.lower, result.upper), (.5, 0., 1.))
        self.assertEqual(result.n_patients, 2)

    def test_constant_cohort_and_reproducibility(self):
        for statistic in ["mean", "median"]:
            first = bootstrap_patient_ci(["a", "b", "c"], [.4] * 3, statistic=statistic, n_resamples=200)
            second = bootstrap_patient_ci(["a", "b", "c"], [.4] * 3, statistic=statistic, n_resamples=200)
            self.assertEqual(first, second)
            self.assertAlmostEqual(first.lower, .4)
            self.assertAlmostEqual(first.upper, .4)

    def test_against_independent_reference(self):
        values = np.array([0., .2, .7, 1.])
        for statistic, reducer in [("mean", np.mean), ("median", np.median)]:
            rng = np.random.default_rng(7)
            indices = rng.integers(0, 4, size=(500, 4))
            expected = np.quantile(reducer(values[indices], axis=1), [.1, .9])
            actual = bootstrap_patient_ci(["a", "b", "c", "d"], values, statistic=statistic,
                                          confidence=.8, n_resamples=500, seed=7)
            np.testing.assert_allclose([actual.lower, actual.upper], expected)
            self.assertEqual(actual.estimate, reducer(values))

    def test_invalid_patient_inputs(self):
        for ids, values in [([], []), (["a"], [1]), (["a", "a"], [0, 1]),
                            (["a", "b"], [1]), (["a", "b"], [0, np.nan]),
                            (["a", "b"], [0, np.inf]), (["a", ""], [0, 1]),
                            (["a", "b"], [[0], [1]])]:
            with self.subTest(ids=ids, values=values), self.assertRaises(ValueError):
                bootstrap_patient_ci(ids, values)
        for kwargs in [{"confidence": 1}, {"confidence": 0}, {"confidence": np.nan},
                       {"n_resamples": 1}, {"n_resamples": 2.5}, {"n_resamples": True},
                       {"statistic": "sum"}, {"seed": -1}, {"seed": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_patient_ci(["a", "b"], [0, 1], **kwargs)


if __name__ == "__main__":
    unittest.main()
