from __future__ import annotations

import jax64  # noqa: F401
import jax.numpy as jnp


def build_dt_sequence(time_grid, dt_default):
    """Construit une suite de pas de temps positifs cohérente avec une grille donnée."""
    dt_default = jnp.asarray(dt_default, dtype=time_grid.dtype)
    diffs = jnp.diff(time_grid, prepend=time_grid[:1])
    diffs = diffs.at[0].set(0.0)
    return jnp.maximum(diffs, 0.0)


def estimate_derivative(time, values, window=5):
    """Estime une dérivée moyenne sur une courte fenêtre initiale."""
    time_arr = jnp.asarray(time, dtype=jnp.float64)
    values_arr = jnp.asarray(values, dtype=jnp.float64)
    n = int(min(window, values_arr.size))
    if n < 2:
        return jnp.asarray(0.0, dtype=jnp.float64)
    span = time_arr[n - 1] - time_arr[0]
    delta = values_arr[n - 1] - values_arr[0]
    safe_span = jnp.where(span == 0.0, 1.0, span)
    deriv = delta / safe_span
    deriv = jnp.where(span == 0.0, 0.0, deriv)
    return jnp.asarray(deriv, dtype=jnp.float64)


__all__ = ["build_dt_sequence", "estimate_derivative"]

