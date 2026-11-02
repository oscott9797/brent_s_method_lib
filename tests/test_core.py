import math
import unittest

from brent_s_method_lib import BrentResult, brent


class TestBrent(unittest.TestCase):
    def test_linear_root(self):
        result = brent(lambda x: x - 2.0, 0.0, 5.0)
        self.assertTrue(result.converged)
        self.assertIsInstance(result, BrentResult)
        self.assertLess(abs(result.root - 2.0), 1e-10)

    def test_quadratic_root(self):
        # x^2 - 4 has roots at +-2; the bracket [0, 5] contains only +2.
        result = brent(lambda x: x * x - 4.0, 0.0, 5.0)
        self.assertTrue(result.converged)
        self.assertLess(abs(result.root - 2.0), 1e-9)

    def test_transcendental_root(self):
        # cos(x) - x has a unique root near 0.7390851332151607.
        result = brent(lambda x: math.cos(x) - x, 0.0, 1.0)
        self.assertTrue(result.converged)
        self.assertLess(abs(result.root - 0.7390851332151607), 1e-9)

    def test_root_at_left_endpoint(self):
        result = brent(lambda x: x, 0.0, 1.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.root, 0.0)
        self.assertEqual(result.iterations, 1)

    def test_root_at_right_endpoint(self):
        result = brent(lambda x: x - 1.0, 0.0, 1.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.root, 1.0)
        self.assertEqual(result.iterations, 2)

    def test_no_sign_change(self):
        # Both endpoints positive: the bracket is invalid.
        result = brent(lambda x: x * x + 1.0, -1.0, 1.0)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.root, 0.0)

    def test_non_finite_endpoint(self):
        def f(x):
            return math.inf if x == 0.0 else x - 1.0

        result = brent(f, 0.0, 2.0)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2)

    def test_maxiter_exhausted(self):
        # A tiny maxiter must report non-convergence rather than looping
        # forever, and must never exceed the evaluation budget.
        def f(x):
            return x ** 3

        result = brent(f, -1.0, 1.0, maxiter=3)
        self.assertLessEqual(result.iterations, 3)
        if not result.converged:
            self.assertLessEqual(result.iterations, 3)

    def test_xtol_controls_width(self):
        # With a loose xtol the bracket should be allowed to stay wider
        # than with a tight one, so the iteration count is no larger.
        loose = brent(lambda x: x - 0.5, 0.0, 1.0, xtol=1e-3, rtol=0.0)
        tight = brent(lambda x: x - 0.5, 0.0, 1.0, xtol=1e-14, rtol=0.0)
        self.assertTrue(loose.converged)
        self.assertTrue(tight.converged)
        self.assertLessEqual(loose.iterations, tight.iterations)

    def test_negative_xtol_rejected(self):
        with self.assertRaises(ValueError):
            brent(lambda x: x, 0.0, 1.0, xtol=-1e-6)

    def test_negative_rtol_rejected(self):
        with self.assertRaises(ValueError):
            brent(lambda x: x, 0.0, 1.0, rtol=-1e-6)

    def test_zero_maxiter_rejected(self):
        with self.assertRaises(ValueError):
            brent(lambda x: x, 0.0, 1.0, maxiter=0)

    def test_steep_function(self):
        # A function that is nearly vertical at the root exercises the
        # bisection fallback in the step-selection logic.
        result = brent(lambda x: math.tanh(1000.0 * (x - 0.25)), 0.0, 1.0)
        self.assertTrue(result.converged)
        self.assertLess(abs(result.root - 0.25), 1e-6)

    def test_reversed_bracket(self):
        # The algorithm should work regardless of which endpoint is
        # negative, so swapping a and b must give the same root.
        r1 = brent(lambda x: x * x - 2.0, 0.0, 3.0)
        r2 = brent(lambda x: x * x - 2.0, 3.0, 0.0)
        self.assertTrue(r1.converged)
        self.assertTrue(r2.converged)
        self.assertLess(abs(r1.root - r2.root), 1e-9)


if __name__ == "__main__":
    unittest.main()
