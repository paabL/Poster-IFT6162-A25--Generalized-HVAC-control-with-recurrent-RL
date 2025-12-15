#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 19:28:00 2025

@author: pablo

"""

from __future__ import annotations

from typing import Any, Mapping
import math

import jax.numpy as jnp

def RC5_steady_state_sys(Ta, Q_solar, Q_con, Q_rad, Qc_dot_val, theta):
    """Résout le steady-state du modèle RC5 (dérivées nulles)."""
    th = theta["th"]
    R_inf, R_w1, R_w2, R_i, R_f, R_c, gA = (
        th[k] for k in ["R_inf", "R_w1", "R_w2", "R_i", "R_f", "R_c", "gA"]
    )
    Q_occ = Q_con + Q_rad

    A = jnp.array([
        [-(1/R_inf+1/R_w2+1/R_f+1/R_i),  1/R_w2,  1/R_i,  1/R_f, 0],
        [ 1/R_w2, -(1/R_w1+1/R_w2),      0,       0,      0],
        [ 1/R_i,  0,                    -1/R_i,   0,      0],
        [ 1/R_f,  0,                     0,     -(1/R_f+1/R_c), 1/R_c],
        [ 0,      0,                     0,       1/R_c, -1/R_c],
    ], dtype=jnp.float64)

    b = jnp.array([
        -Ta/R_inf - gA*Q_solar - Q_occ,
        -Ta/R_w1,
         0.0,
         0.0,
        -Qc_dot_val,
    ], dtype=jnp.float64)

    # Solution libre
    x_free = jnp.linalg.solve(A, b)  # [Tz, Tw, Ti, Tf, Tc]

    # Clip composant par composant dans des plages physiques cohérentes (Kelvin) :
    # - Zone ≈ confort : [15, 30] °C
    # - Parois/masses proches de l'air : [5, 35] °C
    # - Dalle plus inertielle : [15, 30] °C
    # - Fluide condenseur HP : [15, 50] °C
    t_min = 273.15 + jnp.asarray([15.0, 5.0, 15.0, 15.0, 15.0], dtype=jnp.float64)
    t_max = 273.15 + jnp.asarray([30.0, 35.0, 30.0, 40.0, 50.0], dtype=jnp.float64)
    return jnp.clip(x_free, t_min, t_max)


def scale_rc5_building(theta: Mapping[str, Any], k: Mapping[str, float]) -> dict[str, Any]:
    """Scale uniquement les paramètres bâtiment (th), laisse la PAC intacte."""
    th0 = dict(theta.get("th", {}))
    out = dict(theta)
    th = dict(th0)
    out["th"] = th

    k_size = float(k.get("k_size", 1.0))
    k_U = float(k.get("k_U", 1.0))
    k_inf = float(k.get("k_inf", 1.0))
    k_win = float(k.get("k_win", 1.0))
    k_mass = float(k.get("k_mass", 1.0))

    s = math.sqrt(max(k_size, 0.0))
    A_fac = 0.5 * s + 0.5 * k_size

    def mul(key: str, factor: float) -> None:
        if key in th:
            th[key] = th[key] * factor

    mul("C_z", k_size)
    mul("C_w", k_mass * A_fac)
    mul("C_i", k_mass * s)
    mul("C_f", k_mass * k_size)

    mul("R_inf", 1.0 / (max(k_size * k_inf, 1e-12)))

    if "R_w1" in th and "R_w2" in th:
        Rw1_0 = float(th0["R_w1"])
        Rw2_0 = float(th0["R_w2"])
        Rsum0 = Rw1_0 + Rw2_0
        beta = (Rw1_0 / Rsum0) if Rsum0 > 0 else 0.5

        Rsum = Rsum0 / (max(k_U * A_fac, 1e-12))
        th["R_w1"] = th["R_w1"] * 0 + beta * Rsum
        th["R_w2"] = th["R_w2"] * 0 + (1.0 - beta) * Rsum

    mul("R_f", 1.0 / max(k_size, 1e-12))
    mul("R_i", 1.0 / max(s, 1e-12))

    mul("gA", k_size * k_win)

    return out
