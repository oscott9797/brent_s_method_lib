"""Root-finding with Brent's method.

This package exposes :func:`brent` and :class:`BrentResult`.
"""

from .core import BrentResult, brent

__all__ = ["BrentResult", "brent"]
