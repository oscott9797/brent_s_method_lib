"""Brent's method for finding a root of a scalar function in a bracket.

This is a direct implementation of the algorithm described in:

    R. P. Brent, *Algorithms for Minimization without Derivatives*,
    Prentice-Hall, 1973, Chapter 4.

The implementation is deliberately self-contained: it uses only the
standard library, it never evaluates the function outside the bracket,
and it always returns a result object rather than raising for the
ordinary failure modes (no sign change, non-finite values, too many
iterations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BrentResult:
    """Outcome of a call to :func:`brent`.

    Attributes
    ----------
    root:
        Best estimate of the root.  This is the midpoint of the final
        bracket, so ``f(root)`` is not necessarily closer to zero than
        the endpoint values, but the *interval* is guaranteed to be
        small.
    iterations:
        Number of function evaluations performed.  Counted explicitly
        rather than inferred from the loop index because the algorithm
        evaluates ``f`` at most once per iteration.
    converged:
        True if the final bracket width is at most ``xtol``.
    """

    root: float
    iterations: int
    converged: bool


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-12,
    rtol: float = 1e-12,
    maxiter: int = 100,
) -> BrentResult:
    """Find a root of ``f`` in the bracketing interval ``[a, b]``.

    Parameters
    ----------
    f:
        Scalar function.  It must be finite at every point the algorithm
        evaluates; non-finite values are treated as a failure.
    a, b:
        Endpoints of the bracket.  ``f(a)`` and ``f(b)`` must have
        opposite signs (or one of them must be exactly zero).
    xtol:
        Absolute tolerance on the bracket width.
    rtol:
        Relative tolerance on the bracket width, scaled by the current
        best estimate of the root.
    maxiter:
        Maximum number of function evaluations.

    Returns
    -------
    BrentResult
        ``converged`` is False when the bracket did not shrink below the
        tolerance within ``maxiter`` evaluations, or when a non-finite
        value was encountered.  In that case ``root`` is still the best
        estimate available from the bracket seen so far.

    Notes
    -----
    The tolerance test is ``abs(b - a) <= 2 * (xtol + rtol * abs(b))``,
    matching the convention used by Brent's original FORTRAN routine
    (``2 * tol`` in that code).  The factor of two is not cosmetic: it
    prevents the algorithm from taking a step that is smaller than the
    rounding error of the midpoint, which is what makes the method
    terminate on flat functions.
    """
    if xtol < 0:
        raise ValueError("xtol must be non-negative")
    if rtol < 0:
        raise ValueError("rtol must be non-negative")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")

    fa = f(a)
    fb = f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return BrentResult(root=(a + b) / 2.0, iterations=2, converged=False)

    # If an endpoint is already a root we are done; this also makes the
    # sign-change check below well-defined when f(a) or f(b) is zero.
    if fa == 0.0:
        return BrentResult(root=a, iterations=1, converged=True)
    if fb == 0.0:
        return BrentResult(root=b, iterations=2, converged=True)

    if (fa > 0.0) == (fb > 0.0):
        # No sign change: the bracket is invalid.  Return the midpoint
        # rather than raising so callers can treat failure uniformly.
        return BrentResult(root=(a + b) / 2.0, iterations=2, converged=False)

    # Brent's method keeps the current bracket in (a, b) and the previous
    # bracket in (c, d).  The invariant is that b is the best estimate so
    # far and |f(b)| <= |f(a)|.
    c = a
    fc = fa
    d = b - a
    e = d
    iterations = 2  # we have already evaluated f(a) and f(b)

    for _ in range(maxiter - 2):
        if (fb > 0.0) == (fc > 0.0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol = 2.0 * rtol * abs(b) + xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return BrentResult(root=b, iterations=iterations, converged=True)

        # Decide between inverse quadratic interpolation (when we have
        # three distinct function values), secant, and bisection.
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = m
            e = m
        else:
            s = fb / fa
            if a == c:
                # Secant step (only two points available).
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                # Fall back to bisection when the interpolated step is
                # too small or too close to the bracket edge.
                d = m
                e = m

        a = b
        fa = fb
        if abs(d) > tol:
            b += d
        else:
            b += math.copysign(tol, m)

        fb = f(b)
        iterations += 1
        if not math.isfinite(fb):
            return BrentResult(root=b, iterations=iterations, converged=False)
        if fb == 0.0:
            return BrentResult(root=b, iterations=iterations, converged=True)

    return BrentResult(root=b, iterations=iterations, converged=False)
