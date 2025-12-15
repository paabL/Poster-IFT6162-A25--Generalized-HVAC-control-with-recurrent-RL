from __future__ import annotations

import jax64  # noqa: F401
import jax.numpy as jnp
from jax import lax

from Identification.Models import PAC_TAN_K as MODEL_PAC_TAN_K, PAC_TCN_K as MODEL_PAC_TCN_K
from SIMAX.Utils import build_dt_sequence


# ----------------------
# Paramètres RC5 (spécifiques au modèle thermique)
# ----------------------

PARAM_ORDER = [
    ("th", "R_inf"),
    ("th", "R_w2"),
    ("th", "R_f"),
    ("th", "R_i"),
    ("th", "gA"),
    ("th", "C_z"),
    ("th", "R_w1"),
    ("th", "C_w"),
    ("th", "C_i"),
    ("th", "R_c"),
    ("th", "C_f"),
    ("th", "C_c"),
    ("pac", "a_c"),
    ("pac", "b_c"),
    ("pac", "c_c"),
    ("pac", "k_c"),
    ("pac", "a_e"),
    ("pac", "b_e"),
    ("pac", "c_e"),
    ("pac", "k_e"),
]

_DEFAULT_THETA = {
    "th": {
        "R_inf": 5.0e-4,
        "R_w2": 3.0e-4,
        "R_f": 2.0e-4,
        "R_i": 1.0e-4,
        "gA": 2e-3,
        "C_z": 1.5e9,
        "R_w1": 4.0e-4,
        "C_w": 7.5e8,
        "C_i": 2.5e8,
        "R_c": 6.0e-5,
        "C_f": 8.0e8,
        "C_c": 1.0e8,
    },
    "pac": {"a_c": 3500.0, "b_c": -18.0, "c_c": -25.0, "k_c": 1.0},
}

UPDATED_THETA = {
    "th": {
        "R_inf": 2.0e-4,
        "R_w2": 2.0e-5,
        "R_f": 5.0e-5,
        "R_i": 4.0e-5,
        "gA": 10.0,
        "C_z": 3.0e8,
        "R_w1": 2.0e-5,
        "C_w": 5.0e9,
        "C_i": 8.0e9,
        "R_c": 2.0e-5,
        "C_f": 4.0e9,
        "C_c": 1.0e8,
    },
    "pac": {"a_c": 15000.0, "b_c": -200.0, "c_c": 150.0, "k_c": 1.0},
}

_DEFAULT_X0 = jnp.full((5,), 293.15, dtype=jnp.float64)


def copy_theta(template):
    """Convertit un dictionnaire de paramètres en tenseurs float64."""
    return {s: {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()} for s, p in template.items()}


def default_theta():
    """Retourne une copie float64 des paramètres RC par défaut."""
    return copy_theta(_DEFAULT_THETA)


def default_x0():
    """Fournit l'état initial par défaut au format JAX."""
    return jnp.array(_DEFAULT_X0, dtype=jnp.float64)


def theta_from_vec(vec, template=None):
    """Reconstruit le dictionnaire `theta` à partir d'un vecteur plat."""
    vec = jnp.asarray(vec, dtype=jnp.float64)
    if vec.shape != (len(PARAM_ORDER),):
        raise ValueError("Vector length does not match theta dimensionality.")
    template = template or _DEFAULT_THETA
    theta = copy_theta(template)
    for idx, (section, key) in enumerate(PARAM_ORDER):
        theta[section][key] = vec[idx]
    return theta


def pack_params(theta, x0=None):
    """Empile `theta` (et optionnellement `x0`) en un unique vecteur."""
    params = jnp.asarray([theta[s][k] for s, k in PARAM_ORDER], dtype=jnp.float64)
    if x0 is None:
        return params
    x0_vec = jnp.asarray(x0, dtype=jnp.float64)
    return jnp.concatenate((params, x0_vec))


def unpack_params(vec, theta_template=None, x0_template=None):
    """Désempile un vecteur en dictionnaire `theta` et état initial."""
    vec = jnp.asarray(vec, dtype=jnp.float64)
    theta_template = theta_template or _DEFAULT_THETA
    x0_template = (
        jnp.asarray(x0_template, dtype=jnp.float64)
        if x0_template is not None
        else default_x0()
    )
    param_len = len(PARAM_ORDER)
    x_len = x0_template.size
    if vec.size == param_len:
        theta = theta_from_vec(vec, template=theta_template)
        return theta, jnp.array(x0_template, dtype=jnp.float64)
    if vec.size != param_len + x_len:
        raise ValueError("Vector length does not match theta+x0 dimensions.")
    theta = theta_from_vec(vec[:param_len], template=theta_template)
    x0 = jnp.asarray(vec[param_len:], dtype=jnp.float64)
    return theta, x0


def initial_stateRC5_steady(tz, ta, qc, theta):
    """
    Calcule un état initial RC5 à l'équilibre stationnaire (toutes dérivées nulles).
    Utilisé pour l'initialisation BOPTEST en mode online.
    
    Paramètres
    ----------
    tz : float
        Température de zone mesurée [K]
    ta : float
        Température extérieure mesurée [K]
    qc : float
        Puissance condenseur PAC mesurée [W] (via reaQHeaPumCon_y)
    theta : dict
        Paramètres du modèle RC5
    
    Retour
    ------
    jnp.ndarray
        État [Tz, Tw, Ti, Tf, Tc] à l'équilibre [K]
    
    Physique du modèle RC5
    ----------------------
    À l'équilibre stationnaire (ḋot T_i = 0 pour tous les états) :
    
    Équation de T_f (plancher chauffant) :
        C_f·ḋot T_f = (T_z - T_f)/R_f + (T_c - T_f)/R_c = 0
        => (T_f - T_z)/R_f = (T_c - T_f)/R_c = Q̇_floor
    
    Équation de T_c (circuit PAC) :
        C_c·ḋot T_c = (T_f - T_c)/R_c + Q̇_c = 0
        => Q̇_c = (T_c - T_f)/R_c
    
    Donc : Q̇_c mesurée = Q̇_floor = (T_c - T_f)/R_c = (T_f - T_z)/R_f
    
    D'où les formules directes :
        T_f = T_z + Q̇_c · R_f
        T_c = T_f + Q̇_c · R_c
    """
    th = theta["th"]
    
    # 1. Ti à l'équilibre avec Tz (noeud interne capacitif)
    ti = tz

    # 2. Tw à l'équilibre avec Ta et Tz (pont thermique mur)
    # Équation : C_w·ḋot T_w = (T_a - T_w)/R_w1 + (T_z - T_w)/R_w2 = 0
    # => T_w = (T_a·R_w2 + T_z·R_w1) / (R_w1 + R_w2)
    denom_rw = th["R_w1"] + th["R_w2"]
    tw = (ta * th["R_w2"] + tz * th["R_w1"]) / denom_rw

    # 3. Tf et Tc à partir de Qc mesurée
    # Qc (condenseur PAC) = puissance injectée dans le circuit plancher chauffant
    # Flux : Tc -> (R_c) -> Tf -> (R_f) -> Tz
    tf = tz + qc * th["R_f"]  # De l'équation : (T_f - T_z)/R_f = Q̇_c
    tc = tf + qc * th["R_c"]  # De l'équation : (T_c - T_f)/R_c = Q̇_c

    return jnp.array([tz, tw, ti, tf, tc], dtype=jnp.float64)


def initial_stateRC5(sim_data, theta):
    """
    Initialisation classique RC5 à partir de données historiques.
    Utilise la dérivée temporelle de Tz et un bilan thermique complet.
    
    Cette méthode nécessite un historique temporel suffisant (≥2 points)
    et estime les états cachés (Tw, Ti, Tf, Tc) via inversion du modèle
    et résolution polynomiale pour Tc.
    
    Paramètres
    ----------
    sim_data : Sim_and_Data
        Données de simulation avec historique temporel
    theta : dict
        Paramètres du modèle RC5
    
    Retour
    ------
    jnp.ndarray
        État initial [Tz, Tw, Ti, Tf, Tc] [K]
    """
    sim = sim_data.simulation
    data = sim_data.dataset

    th, pac = theta["th"], theta["pac"]

    tz_series = jnp.asarray(data.d["reaTZon_y"], dtype=jnp.float64)
    qc_series = jnp.asarray(data.d["reaQHeaPumCon_y"], dtype=jnp.float64)
    tz0 = tz_series[0]
    ta0 = jnp.asarray(data.d["weaSta_reaWeaTDryBul_y"][0], dtype=jnp.float64)
    qc0 = qc_series[0]

    time = jnp.asarray(data.time, dtype=jnp.float64)
    
    # Tentative d'initialisation dynamique (si historique suffisant)
    try:
        if len(time) < 2:
            raise ValueError("Pas assez de données pour estimer la dérivée.")
            
        from SIMAX.Simulation import Sim_and_Data as _SimData
        d_tz0 = _SimData.estimate_derivative(time, tz_series)
        
        qrad0 = jnp.asarray(data.d["weaSta_reaWeaHGloHor_y"][0], dtype=jnp.float64)
        qocc0 = jnp.asarray(data.d["InternalGainsCon[1]"][0] + data.d["InternalGainsRad[1]"][0], dtype=jnp.float64)

        signals = getattr(sim.controller, "signals", None)
        if signals is not None and "oveHeaPumY_u" in signals:
            u_hp0 = jnp.asarray(signals["oveHeaPumY_u"][0], dtype=jnp.float64)
        else:
            u_hp0 = jnp.asarray(0.0, dtype=jnp.float64)

        denom_rw = th["R_w1"] + th["R_w2"]
        tw0 = (th["R_w2"] * ta0 + th["R_w1"] * tz0) / denom_rw
        ti0 = tz0
        
        # Méthode classique : estimation de Tf via bilan thermique de la zone
        # À partir de l'équation : C_z·ḋot T_z = ... + (T_f - T_z)/R_f + ...
        # On isole : (T_f - T_z)/R_f = C_z·ḋot T_z - [autres flux]
        heat_balance = th["C_z"] * d_tz0 - (ta0 - tz0) / th["R_inf"] - (ta0 - tz0) / denom_rw - th["gA"] * qrad0 - qocc0
        tf0 = tz0 + th["R_f"] * heat_balance

        # Estimation de Tc via inversion du modèle PAC
        k_c = pac["k_c"]
        scale = k_c * u_hp0
        cond = jnp.abs(scale) > 1e-8
        T_cn = pac.get("Tcn", jnp.asarray(MODEL_PAC_TCN_K, dtype=jnp.float64))
        T_an = pac.get("Tan", jnp.asarray(MODEL_PAC_TAN_K, dtype=jnp.float64))
        tc0_inactive = tf0 + th["R_c"] * qc0

        def active_branch(_):
            return T_cn + ((qc0 / scale - pac["a_c"] - pac["c_c"] * (ta0 - T_an)) / pac["b_c"])

        def inactive_branch(_):
            return tc0_inactive

        tc0 = lax.cond(cond, active_branch, inactive_branch, operand=None)
        
        x0 = jnp.array([tz0, tw0, ti0, tf0, tc0], dtype=jnp.float64)
        
        # Vérification de cohérence (températures aberrantes)
        if jnp.any(x0 > 373.15) or jnp.any(x0 < 223.15) or not jnp.all(jnp.isfinite(x0)):
             raise ValueError("État initial dynamique aberrant.")
             
        return x0

    except Exception:
        # Fallback : Initialisation stationnaire basée sur Qc mesuré
        return initial_stateRC5_steady(tz0, ta0, qc0, theta)


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

    return jnp.linalg.solve(A, b)  # [Tz, Tw, Ti, Tf, Tc]


__all__ = [
    "PARAM_ORDER",
    "default_theta",
    "default_x0",
    "theta_from_vec",
    "pack_params",
    "unpack_params",
    "build_dt_sequence",
    "initial_stateRC5",
    "initial_stateRC5_steady",
]
