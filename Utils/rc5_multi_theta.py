import numpy as np
import gymnasium as gym
from pathlib import Path

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import gymRC5 as rc5
from Utils.utils import scale_rc5_building


_THETA_UNITS = {
    "th": {
        "R_inf": "K/W",
        "R_w1": "K/W",
        "R_w2": "K/W",
        "R_f": "K/W",
        "R_i": "K/W",
        "R_c": "K/W",
        "gA": "m^2",
        "C_z": "J/K",
        "C_w": "J/K",
        "C_f": "J/K",
        "C_c": "J/K",
        "C_i": "J/K",
    },
    "pac": {
        "a_c": "W",
        "b_c": "W/K",
        "c_c": "W/K",
        "k_c": "",
        "a_e": "W",
        "b_e": "W/K",
        "c_e": "W/K",
        "k_e": "",
    },
}


def _collect_entries(base_dict, curr_dict):
    entries = []
    if not (isinstance(base_dict, dict) and isinstance(curr_dict, dict)):
        return entries
    for section, sect_vals in base_dict.items():
        if not isinstance(sect_vals, dict):
            continue
        curr_sect = curr_dict.get(section, {}) if isinstance(curr_dict, dict) else {}
        for key, b_val in sect_vals.items():
            b = float(np.asarray(b_val))
            c = float(np.asarray(curr_sect.get(key, b))) if isinstance(curr_sect, dict) else b
            rel = (c - b) / b if b != 0.0 else 0.0
            unit = _THETA_UNITS.get(section, {}).get(key, "")
            name = f"{section}.{key}"
            entries.append((name, c, rel, unit))
    return entries


def _export_thetas_markdown(base_theta, thetas):
    lines = [
        "# Paramètres des modèles RC5",
        "",
        f"Nombre de modèles : {len(thetas)}",
        "",
    ]

    for idx, theta_i in enumerate(thetas):
        entries = _collect_entries(base_theta, theta_i)
        header = f"## Modèle {idx}"
        lines.append(header)
        lines.append("")
        lines.append("| Paramètre | Valeur | Δ rel. vs base | Unité |")
        lines.append("|-----------|--------|----------------|-------|")
        for name, c, rel, unit in entries:
            rel_str = f"{rel:+.1%}"
            unit_str = unit if unit else "-"
            lines.append(f"| `{name}` | {c:.6g} | {rel_str} | {unit_str} |")
        lines.append("")

    md_path = Path("theta_parameters.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_thetas(n_buildings: int = 10, noise_level: float = 0.05, seed: int = 0):
    """Crée une liste de thetas légèrement perturbés autour du modèle identifié."""
    base_theta = rc5.sim_opti_loaded.model.theta
    # On ne perturbe que la partie thermique "th", la PAC reste fixe.
    base_th = base_theta["th"]
    flat, unravel = ravel_pytree(base_th)
    flat = np.asarray(flat, dtype=np.float64)

    rng = np.random.default_rng(seed)
    thetas = []
    for i in range(n_buildings):
        if i == 0:
            thetas.append(base_theta)
        else:
            eps = rng.normal(loc=0.0, scale=noise_level, size=flat.shape)
            th_i = unravel(jnp.asarray(flat * (1.0 + eps), dtype=jnp.float64))
            theta_i = {"th": th_i, "pac": base_theta["pac"]}
            thetas.append(theta_i)
    # Export Markdown une fois pour toutes à la création des modèles
    _export_thetas_markdown(base_theta, thetas)
    return thetas


def build_k_models(
    ks: list[dict[str, float]],
    *,
    base_theta=None,
):
    base = base_theta or rc5.sim_opti_loaded.model.theta
    return [scale_rc5_building(base, k) for k in ks]


class RandomThetaWrapper(gym.Wrapper):
    """Tire un theta à chaque reset et l'injecte dans env.theta."""

    def __init__(self, env, thetas, seed: int = 0, fixed_building_idx: int | None = None):
        super().__init__(env)
        self.thetas = list(thetas)
        self.fixed_building_idx = fixed_building_idx
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        idx = int(self.fixed_building_idx) if self.fixed_building_idx is not None else int(self.rng.integers(0, len(self.thetas)))
        theta = self.thetas[idx]
        self.env.theta = theta
        self.env.theta_idx = idx
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {**dict(info), "theta_idx": idx, "theta": theta}


class KModelWrapper(gym.Wrapper):
    """Sélectionne un modèle (k, theta) pré-construit par épisode (random ou fixe)."""

    def __init__(
        self,
        env,
        *,
        thetas: list[dict],
        ks: list[dict[str, float]] | None = None,
        seed: int = 0,
        fixed_model_idx: int | None = None,
    ):
        super().__init__(env)
        self.fixed_model_idx = fixed_model_idx
        self.rng = np.random.default_rng(seed)
        self.thetas = list(thetas)
        self.ks = list(ks) if ks is not None else None

    def reset(self, *, seed=None, options=None):
        if self.fixed_model_idx is not None:
            idx = int(self.fixed_model_idx)
        else:
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            idx = int(self.rng.integers(0, len(self.thetas)))
        self.env.theta = self.thetas[idx]
        self.env.theta_idx = idx
        if self.ks is not None:
            self.env.k = self.ks[idx]
        obs, info = self.env.reset(seed=seed, options=options)
        out = {**dict(info), "theta_idx": idx, "theta": self.thetas[idx]}
        if self.ks is not None:
            out["k"] = self.ks[idx]
        return obs, out


RandomKWrapper = KModelWrapper
