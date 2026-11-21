import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze import fit_window, GPA


class FitChecks(unittest.TestCase):
    def test_known_quadratic_stiffness_and_minimum(self):
        v = np.linspace(18, 22, 17)
        e = -5 + .5 * (100 / GPA) / 20 * (v - 20)**2
        fit = fit_window(v, e, 2)
        self.assertTrue(fit['valid'])
        self.assertAlmostEqual(fit['stiffness_GPa'], 100, places=6)
        self.assertAlmostEqual(fit['volume_A3_atom'], 20, places=6)

    def test_boundary_minimum_not_extrapolated(self):
        v = np.linspace(18, 22, 17)
        self.assertFalse(fit_window(v, v**2, 2)['valid'])


if __name__ == '__main__':
    unittest.main()
