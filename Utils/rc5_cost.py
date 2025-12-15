from __future__ import annotations

from typing import Any

import numpy as np


# "Euro" weights used in gymRC5 (reward = -cost).
DEFAULT_W_ENERGY_EUR = 1.0              # €/€  (energy cost already in euros)
DEFAULT_W_COMFORT_EUR_PER_KH = 5.0      # €/K·h
DEFAULT_W_SAT_EUR_PER_UNIT_H = 0.2      # €/(|delta_sat|·h)


def interval_components(  # noqa: PLR0913
    *,
    t_step_s: Any,
    tz_seq_k: Any,
    lower_seq_k: Any,
    upper_seq_k: Any,
    occ_seq: Any,
    php_w_seq: Any,
    price_seq: Any,
    comfort_huber_k: float = 0.0,
    u_seq: Any | None = None,
    delta_sat_seq: Any | None = None,
    xp=np,
):
    """
    Compute cost components over intervals ("gymRC5" alignment).

    Conventions (compatible with gymRC5.py and MPC_RC5-RC5.py):
    - t_step_s : integration point times (in seconds), shape N
    - tz_seq_k : Tz "on intervals" (often tz[1:]), shape N
    - lower/upper/occ/price/php : aligned signals, shape N

    Returns:
    - energy_eur : ∫ price(€/kWh) * max(php,0)(kW) dt(h)  [€]
    - comfort_kh : ∫ (violation(K) * occ) dt(h)          [K·h]
      Option: soften the violation near 0 with a Huber (comfort_huber_k > 0)
        v_eff = 0.5*v^2/delta  if v <= delta
                v - 0.5*delta  else
    - u_unit_h   : ∫ |u| dt(h)                           [unit·h] (0 if u_seq=None)
    - tz_kh      : ∫ Tz(K) dt(h)                         [K·h]
    - sat_unit_h : ∫ |delta_sat| dt(h)                   [unit·h] (0 if delta_sat_seq=None)
    """
    t_step_s = xp.asarray(t_step_s, dtype=xp.float64)
    if t_step_s.size < 2:
        zero = xp.asarray(0.0, dtype=xp.float64)
        return zero, zero, zero, zero, zero

    tz_seq_k = xp.asarray(tz_seq_k, dtype=xp.float64)
    lower_seq_k = xp.asarray(lower_seq_k, dtype=xp.float64)
    upper_seq_k = xp.asarray(upper_seq_k, dtype=xp.float64)
    occ_seq = xp.asarray(occ_seq, dtype=xp.float64)
    php_w_seq = xp.asarray(php_w_seq, dtype=xp.float64)
    price_seq = xp.asarray(price_seq, dtype=xp.float64)

    comfort_dev = xp.maximum(lower_seq_k - tz_seq_k, 0.0) + xp.maximum(tz_seq_k - upper_seq_k, 0.0)
    if float(comfort_huber_k) > 0.0:
        # v_eff has the same unit as v (Kelvin), but is smoother near 0.
        delta = xp.asarray(comfort_huber_k, dtype=xp.float64)
        comfort_dev = xp.where(
            comfort_dev <= delta,
            0.5 * (comfort_dev * comfort_dev) / delta,
            comfort_dev - 0.5 * delta,
        )
    comfort_kh = xp.trapezoid(comfort_dev * occ_seq, x=t_step_s) / 3600.0

    php_pos_kw = xp.maximum(php_w_seq, 0.0) / 1000.0
    energy_eur = xp.trapezoid(price_seq * php_pos_kw, x=t_step_s / 3600.0)

    if u_seq is None:
        u_unit_h = xp.asarray(0.0, dtype=xp.float64)
    else:
        u_seq = xp.asarray(u_seq, dtype=xp.float64)
        u_unit_h = xp.trapezoid(xp.abs(u_seq), x=t_step_s) / 3600.0

    tz_kh = xp.trapezoid(tz_seq_k, x=t_step_s) / 3600.0

    if delta_sat_seq is None:
        sat_unit_h = xp.asarray(0.0, dtype=xp.float64)
    else:
        delta_sat_seq = xp.asarray(delta_sat_seq, dtype=xp.float64)
        sat_unit_h = xp.trapezoid(xp.abs(delta_sat_seq), x=t_step_s) / 3600.0

    return energy_eur, comfort_kh, u_unit_h, tz_kh, sat_unit_h


def interval_reward_and_terms(  # noqa: PLR0913
    *,
    t_step_s: Any,
    tz_seq_k: Any,
    lower_seq_k: Any,
    upper_seq_k: Any,
    occ_seq: Any,
    php_w_seq: Any,
    price_seq: Any,
    u_seq: Any | None = None,
    delta_sat_seq: Any | None = None,
    w_energy: float = DEFAULT_W_ENERGY_EUR,
    w_comfort: float = DEFAULT_W_COMFORT_EUR_PER_KH,
    comfort_huber_k: float = 0.0,
    w_sat: float = DEFAULT_W_SAT_EUR_PER_UNIT_H,
    w_u: float = 0.0,
    w_tz: float = 0.0,
    xp=np,
):
    """
    GymRC5-compatible reward in euros: reward = -(wE*energy + wC*comfort + wS*sat + wU*∫|u| + wTz*∫Tz).

    Returns (reward, (comfort_term, energy_term, sat_term)) where each term is negative.
    """
    energy_eur, comfort_kh, u_unit_h, tz_kh, sat_unit_h = interval_components(
        t_step_s=t_step_s,
        tz_seq_k=tz_seq_k,
        lower_seq_k=lower_seq_k,
        upper_seq_k=upper_seq_k,
        occ_seq=occ_seq,
        php_w_seq=php_w_seq,
        price_seq=price_seq,
        comfort_huber_k=comfort_huber_k,
        u_seq=u_seq,
        delta_sat_seq=delta_sat_seq,
        xp=xp,
    )

    comfort_term = -xp.asarray(w_comfort, dtype=xp.float64) * comfort_kh
    energy_term = -xp.asarray(w_energy, dtype=xp.float64) * energy_eur
    sat_term = -xp.asarray(w_sat, dtype=xp.float64) * sat_unit_h
    u_term = -xp.asarray(w_u, dtype=xp.float64) * u_unit_h
    tz_term = -xp.asarray(w_tz, dtype=xp.float64) * tz_kh
    reward = comfort_term + energy_term + sat_term + u_term + tz_term

    return reward, (comfort_term, energy_term, sat_term)
