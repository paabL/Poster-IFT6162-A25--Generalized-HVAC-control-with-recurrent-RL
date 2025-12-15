from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import jax64  # noqa: F401
import jax.numpy as jnp

from SIMAX.Controller import Controller_constSeq
from SIMAX.Simulation import Simulation_JAX, SimulationDataset, Sim_and_Data


@dataclass(frozen=True)
class ValidationState:
    """Résultat d'une validation : sorties simulées et métriques calculées."""

    theta: dict[str, Any]
    time: jnp.ndarray
    tz_sim: jnp.ndarray
    qc_sim: jnp.ndarray
    qe_sim: jnp.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ValidationBase(ABC):
    """Super-classe minimale pour valider un modèle à partir d'un dataset."""

    simulation: Simulation_JAX
    dataset: SimulationDataset
    state: ValidationState | None = field(default=None, init=False)

    def simulation_for_dataset(self, dataset):
        """dataset (SimulationDataset) -> Simulation_JAX. Construit une copie de la simulation courante adaptée au dataset fourni."""
        return self.simulation.copy(time_grid=dataset.time, d=dataset.d)

    def run(self, theta=None):
        """theta (dict[str, Any] | None) -> ValidationState. Lance la simulation et calcule les métriques associées."""
        sim = self.simulation_for_dataset(self.dataset)
        time, y_sim, *_ = sim.run(theta)
        y_arr = jnp.asarray(y_sim, dtype=jnp.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        if y_arr.shape[1] < 3:
            raise ValueError("ValidationBase.run suppose au moins 3 sorties (tz, qc, qe).")
        tz_sim = y_arr[:, 0]
        qc_sim = y_arr[:, 1]
        qe_sim = y_arr[:, 2]
        theta_used = theta if theta is not None else sim.model.theta
        metrics = self.compute_metrics(tz_sim, qc_sim, qe_sim)
        object.__setattr__(
            self,
            "state",
            ValidationState(
                theta=theta_used,
                time=time,
                tz_sim=tz_sim,
                qc_sim=qc_sim,
                qe_sim=qe_sim,
                metrics=metrics,
            ),
        )
        return self.state

    def report(self, state=None):
        """state (ValidationState | None) -> dict[str, Any]. Retourne les métriques calculées pour inspection rapide."""
        current = state if state is not None else self.state
        if current is None:
            raise RuntimeError("Validation not run yet; call run() before report().")
        return dict(current.metrics)

    def plot(self, state=None, *, theta=None, path=None):
        """state (ValidationState | None), theta (dict[str, Any] | None), path (str | None) -> None. Trace les séries simulées vs mesures via Sim_and_Data."""
        current = state if state is not None else self.state
        theta_used = theta
        if theta_used is None:
            if current is None:
                raise RuntimeError("Validation not run yet; provide theta or call run().")
            theta_used = current.theta
        sim = self.simulation_for_dataset(self.dataset)
        # Pour RC5, on empile les trois sorties mesurées en vecteur y_meas
        d = self.dataset.d
        y_meas = jnp.stack((d["reaTZon_y"], d["reaQHeaPumCon_y"], d["reaQHeaPumEva_y"]), axis=-1)
        sim_data = Sim_and_Data(simulation=sim, dataset=self.dataset, y_meas=y_meas, W=None, initial_state_fn=None)
        sim_data.plot(theta=theta_used, path=path)

    @abstractmethod
    def compute_metrics(self, tz_sim, qc_sim, qe_sim):
        """tz_sim (jnp.ndarray), qc_sim (jnp.ndarray), qe_sim (jnp.ndarray) -> dict[str, Any]. Calcule les métriques d'évaluation entre simulation et mesures."""
        ...


class Validation_RC5(ValidationBase):
    """Validation élémentaire des sorties Tz/Qc_dot/Qe_dot du modèle RC5."""

    def simulation_for_dataset(self, dataset):  # type: ignore[override]
        """dataset (SimulationDataset) -> Simulation_JAX. Adapter la simulation RC5 au dataset fourni."""
        base = super().simulation_for_dataset(dataset)
        controller = base.controller
        if isinstance(controller, Controller_constSeq):
            controller = Controller_constSeq(oveHeaPumY_u=dataset.u["oveHeaPumY_u"])
        return base.copy(controller=controller)

    def compute_metrics(self, tz_sim, qc_sim, qe_sim):  # type: ignore[override]
        """tz_sim (jnp.ndarray), qc_sim (jnp.ndarray), qe_sim (jnp.ndarray) -> dict[str, Any]. Compare les sorties RC5 aux mesures."""

        def stats(err, prefix):
            """err (jnp.ndarray), prefix (str) -> dict[str, float]. Calcule RMSE/MAE/biais pour un signal."""
            rmse = float(jnp.sqrt(jnp.mean(err**2)))
            mae = float(jnp.mean(jnp.abs(err)))
            bias = float(jnp.mean(err))
            return {f"{prefix}_rmse": rmse, f"{prefix}_mae": mae, f"{prefix}_bias": bias}

        d = self.dataset.d
        tz_meas = d["reaTZon_y"]
        qc_meas = d["reaQHeaPumCon_y"]
        qe_meas = d["reaQHeaPumEva_y"]
        tz_err = tz_sim - tz_meas
        qc_err = qc_sim - qc_meas
        qe_err = qe_sim - qe_meas
        metrics: dict[str, float] = {}
        for prefix, err in (("tz", tz_err), ("qc", qc_err), ("qe", qe_err)):
            metrics.update(stats(err, prefix))
        return metrics


__all__ = [
    "ValidationState",
    "ValidationBase",
    "Validation_RC5",
]
