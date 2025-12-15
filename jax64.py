from __future__ import annotations

import types
import numpy as _np

# Some JAX builds expect numpy.dtypes (added in numpy >=2.0); provide a stub for older numpy.
if not hasattr(_np, "dtypes"):
    _np.dtypes = types.SimpleNamespace()

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
