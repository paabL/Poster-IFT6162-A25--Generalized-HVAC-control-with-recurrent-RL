from __future__ import annotations

import jax64  # noqa: F401
import jax.numpy as jnp

from SIMAX.Models import Model_JAX

PAC_TAN_K = 273.15 + 7.0
PAC_TCN_K = 273.15 + 35.0


#1 - RC5

def Qc_dot(Tc, Ta, u_hp, pac):
    """Puissance condensateur de la PAC.

    Paramètres
    ----------
    Tc : jnp.ndarray
        Température côté condenseur [K].
    Ta : jnp.ndarray
        Température d'air extérieur [K].
    u_hp : jnp.ndarray
        Signal de modulation de la PAC [-].
    pac : dict
        Paramètres de la carte PAC (a_c, b_c, c_c, k_c, etc.).

    Retour
    ------
    jnp.ndarray
        Puissance thermique délivrée au condenseur [W].
    """
    T_cn = pac.get("Tcn", PAC_TCN_K)
    T_an = pac.get("Tan", PAC_TAN_K)
    poly = pac["a_c"] + pac["b_c"] * (Tc - T_cn) + pac["c_c"] * (Ta - T_an)
    qc = pac["k_c"] * poly * u_hp
    return qc


def Qe_dot(Tc, Ta, u_hp, pac):
    """Puissance évaporateur de la PAC (signe opposé à Qc_dot).

    Paramètres
    ----------
    Tc : jnp.ndarray
        Température côté condenseur [K].
    Ta : jnp.ndarray
        Température d'air extérieur [K].
    u_hp : jnp.ndarray
        Signal de modulation de la PAC [-].
    pac : dict
        Paramètres de la carte PAC.

    Retour
    ------
    jnp.ndarray
        Puissance thermique absorbée à l'évaporateur [W] (signée négativement).
    """
    T_cn = pac.get("Tcn", PAC_TCN_K)
    T_an = pac.get("Tan", PAC_TAN_K)
    poly = pac["a_e"] + pac["b_e"] * (Tc - T_cn) + pac["c_e"] * (Ta - T_an)
    qe = pac["k_e"] * poly * u_hp
    return -qe


def RC5_state_derivative(state, theta, Ta, Q_solar, Q_con, Q_rad, u_hp):
    """Dynamique continue du modèle RC5.

    Paramètres
    ----------
    state : jnp.ndarray
        État thermique [Tz, Tw, Ti, Tf, Tc] en K.
    theta : dict
        Paramètres thermiques et PAC (sous-dictionnaires 'th' et 'pac').
    Ta : jnp.ndarray
        Température extérieure [K].
    Q_solar : jnp.ndarray
        Flux solaire global incident [W].
    Q_con : jnp.ndarray
        Gains internes convectifs [W].
    Q_rad : jnp.ndarray
        Gains internes radiatifs [W].
    u_hp : jnp.ndarray
        Signal de modulation de la PAC [-].

    Retour
    ------
    jnp.ndarray
        Dérivée temporelle de l'état (dTz/dt, dTw/dt, dTi/dt, dTf/dt, dTc/dt).
    """
    th = theta["th"]
    pac = theta["pac"]
    Tz, Tw, Ti, Tf, Tc = state
    Q_occ = Q_con + Q_rad
    Qc_dot_val = Qc_dot(Tc, Ta, u_hp, pac)
    dTz = (
        (Ta - Tz) / th["R_inf"]
        + (Tw - Tz) / th["R_w2"]
        + (Tf - Tz) / th["R_f"]
        + (Ti - Tz) / th["R_i"]
        + th["gA"] * Q_solar
        + Q_occ
    ) / th["C_z"]
    dTw = ((Ta - Tw) / th["R_w1"] + (Tz - Tw) / th["R_w2"]) / th["C_w"]
    dTi = ((Tz - Ti) / th["R_i"]) / th["C_i"]
    dTf = ((Tz - Tf) / th["R_f"] + (Tc - Tf) / th["R_c"]) / th["C_f"]
    dTc = ((Tf - Tc) / th["R_c"] + Qc_dot_val) / th["C_c"]
    return jnp.array([dTz, dTw, dTi, dTf, dTc])


def rc5_state_fn(x, u, d, theta):
    """Wrapper générique RC5 pour `Model_JAX.state_fn`.

    Paramètres
    ----------
    x : jnp.ndarray
        État thermique [Tz, Tw, Ti, Tf, Tc] en K.
    u : Mapping[str, Any]
        Commandes de contrôle (contient typiquement 'oveHeaPumY_u').
    d : Mapping[str, Any]
        Perturbations (Ta, Q_solar, gains internes, ...).
    theta : dict
        Paramètres du modèle RC5.

    Retour
    ------
    jnp.ndarray
        Dérivée continue de l'état.
    """
    u_hp = jnp.asarray(u["oveHeaPumY_u"])
    Q_con = jnp.asarray(d["InternalGainsCon[1]"])
    Q_rad = jnp.asarray(d["InternalGainsRad[1]"])
    Q_solar = jnp.asarray(d["weaSta_reaWeaHGloHor_y"])
    Ta = jnp.asarray(d["weaSta_reaWeaTDryBul_y"])
    return RC5_state_derivative(jnp.asarray(x), theta, Ta, Q_solar, Q_con, Q_rad, u_hp)


def rc5_qc_dot(x, u, d, theta):
    """Puissance condensateur vue depuis l'état x (wrapper pour Qc_dot)."""
    state = jnp.asarray(x)
    return Qc_dot(
        state[-1],  # Tc
        jnp.asarray(d["weaSta_reaWeaTDryBul_y"]),
        jnp.asarray(u["oveHeaPumY_u"]),
        theta["pac"],
    )


def rc5_qe_dot(x, u, d, theta):
    """Puissance évaporateur vue depuis l'état x (wrapper pour Qe_dot)."""
    state = jnp.asarray(x)
    return Qe_dot(
        state[-1],  # Tc
        jnp.asarray(d["weaSta_reaWeaTDryBul_y"]),
        jnp.asarray(u["oveHeaPumY_u"]),
        theta["pac"],
    )


def rc5_output_fn(x, u, d, theta):
    """Fonction d'observation RC5 : renvoie (Tz, Qc_dot, Qe_dot).

    Paramètres
    ----------
    x : jnp.ndarray
        État thermique [Tz, Tw, Ti, Tf, Tc] en K.
    u : Mapping[str, Any]
        Commandes de contrôle.
    d : Mapping[str, Any]
        Perturbations.
    theta : dict
        Paramètres du modèle.

    Retour
    ------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Température de zone [K], puissance condenseur [W], puissance évaporateur [W].
    """
    state = jnp.asarray(x)
    tz = state[0]
    qc = rc5_qc_dot(state, u, d, theta)
    qe = rc5_qe_dot(state, u, d, theta)
    return tz, qc, qe


RC5_STATE_NAMES = ("Tz", "Tw", "Ti", "Tf", "Tc")
RC5_STATE_UNITS = ("K", "K", "K", "K", "K")
RC5_OUTPUT_NAMES = ("Tz", "Qc_dot", "Qe_dot")
RC5_OUTPUT_UNITS = ("K", "W", "W")
RC5_CONTROL_NAMES = ("oveHeaPumY_u",)
RC5_CONTROL_UNITS = ("-",)
RC5_DISTURBANCE_NAMES = (
    "weaSta_reaWeaTDryBul_y",
    "weaSta_reaWeaHGloHor_y",
    "InternalGainsCon[1]",
    "InternalGainsRad[1]",
)
RC5_DISTURBANCE_UNITS = ("K", "W", "W", "W")
