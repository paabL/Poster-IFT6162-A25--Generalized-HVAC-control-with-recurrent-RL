from __future__ import annotations

import equinox as eqx
import jax64  # noqa: F401
import jax
import jax.numpy as jnp




class Model_JAX(eqx.Module):
    """Modèle dynamique générique basé sur JAX.

    Ce conteneur encapsule :
    - une dynamique continue `state_fn(x, u, d, theta) -> dx/dt`,
    - une fonction d'observation `output_fn(x, u, d, theta) -> y`,
    - et des métadonnées (noms/unités) décrivant états, sorties, commandes et perturbations.
    """

    theta: dict
    state_fn: callable
    output_fn: callable

    # Métadonnées optionnelles :
    # - state_*      : états internes x
    # - output_*     : sorties de h(x, u, d)
    # - control_*    : entrées de commande
    # - disturbance_*: perturbations d
    state_names: tuple[str, ...] | None = None
    state_units: tuple[str, ...] | None = None
    output_names: tuple[str, ...] | None = None
    output_units: tuple[str, ...] | None = None
    control_names: tuple[str, ...] | None = None
    control_units: tuple[str, ...] | None = None
    disturbance_names: tuple[str, ...] | None = None
    disturbance_units: tuple[str, ...] | None = None

    def state_derivative(self, x, u, d):
        """Évalue dx/dt pour un état et des entrées donnés."""
        return self.state_fn(jnp.asarray(x), u, d, self.theta)

    def h(self, x, u, d):
        """Calcule la sortie du modèle pour l'état et les entrées fournis."""
        return self.output_fn(jnp.asarray(x), u, d, self.theta)


__all__ = ["Model_JAX", "PAC_TAN_K", "PAC_TCN_K"]

