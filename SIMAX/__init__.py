from __future__ import annotations

"""SIMAX: generic JAX simulation core.

Concrete modules are organized by role:
- `SIMAX.Models`      : generic JAX model container;
- `SIMAX.Simulation`  : simulation engine and identification tools;
- `SIMAX.Controller`  : generic JAX-compatible controllers;
- `SIMAX.Params`      : JAX parameter utilities.

Project-specific code (RC5, BOPTEST, application scripts) stays at the repo root.
"""

import jax64  # noqa: F401

__all__: list[str] = []
