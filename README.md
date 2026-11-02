# brent_s_method_lib

Finds a root of a scalar function in a bracketing interval without derivatives, combining bisection, secant, and inverse quadratic interpolation.

```python
from brent_s_method_lib import brent

result = brent(lambda x: x * x - 2.0, 0.0, 3.0)
print(result.root)        # ~1.414213562
print(result.converged)   # True
print(result.iterations)  # number of function evaluations
```

The public API is exactly two names:

- `brent(f, a, b, *, xtol=1e-12, rtol=1e-12, maxiter=100) -> BrentResult`
- `BrentResult`, a frozen dataclass with fields `root`, `iterations`, and `converged`.

## Why this exists

Bisection is robust but slow; the secant method is fast but can leave the
bracket. Brent's method gets both: it keeps a bracket at all times and
uses inverse quadratic interpolation or a secant step whenever those are
safe, falling back to bisection when they are not. The trade-off is that
the algorithm is fiddly — the step-selection logic has several
conditions that exist only to keep the iteration bounded, and getting
them wrong produces a method that is either slower than bisection or
that wanders outside the bracket. This library keeps the original
Brent conditions rather than simplifying them, so the iteration count
matches the published algorithm.

## Awkward edge

The bracket must actually bracket: `f(a)` and `f(b)` must have opposite
signs. If they do not, `brent` returns a `BrentResult` with
`converged=False` instead of raising, so a caller can treat a bad
bracket the same way as a non-converged run. The same applies when the
function returns a non-finite value at an evaluation point. If you want
a hard failure on a bad bracket, check `converged` yourself and raise.

One more thing worth knowing: the convergence test uses the bracket
width scaled by `2 * (xtol + rtol * abs(b))`, matching Brent's original
FORTRAN routine. The factor of two is deliberate — it stops the method
from taking a step smaller than the rounding error of the midpoint,
which is what allows it to terminate on functions with a flat region
near the root.
