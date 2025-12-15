from __future__ import annotations

"""SIMAX : noyau générique de simulation JAX.

Les modules concrets sont organisés par rôle :
- `SIMAX.Models`      : conteneur de modèle JAX générique ;
- `SIMAX.Simulation`  : moteur de simulation et outils d'identification ;
- `SIMAX.Controller`  : contrôleurs génériques compatibles JAX ;
- `SIMAX.Params`      : utilitaires de paramètres JAX.

Le code spécifique (RC5, BOPTEST, scripts d'application) reste à la racine.
"""

import jax64  # noqa: F401

__all__: list[str] = []

