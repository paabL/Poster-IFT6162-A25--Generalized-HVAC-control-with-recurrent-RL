from __future__ import annotations

import jax64  # noqa: F401
import jax.numpy as jnp

from SIMAX.Models import Model_JAX

PAC_TAN_K = 273.15 + 7.0
PAC_TCN_K = 273.15 + 35.0


#1 - RC5

def Qc_dot(Tc, Ta, u_hp, pac):
    """Heat pump condenser power.

    Parameters
    ----------
    Tc : jnp.ndarray
        Condenser-side temperature [K].
    Ta : jnp.ndarray
        Outdoor air temperature [K].
    u_hp : jnp.ndarray
        Heat pump modulation signal [-].
    pac : dict
        Heat pump map parameters (a_c, b_c, c_c, k_c, etc.).

    Returns
    ------
    jnp.ndarray
        Thermal power delivered at the condenser [W].
    """
    T_cn = pac.get("Tcn", PAC_TCN_K)
    T_an = pac.get("Tan", PAC_TAN_K)
    poly = pac["a_c"] + pac["b_c"] * (Tc - T_cn) + pac["c_c"] * (Ta - T_an)
    qc = pac["k_c"] * poly * u_hp
    return qc


def Qe_dot(Tc, Ta, u_hp, pac):
    """Heat pump evaporator power (opposite sign of Qc_dot).

    Parameters
    ----------
    Tc : jnp.ndarray
        Condenser-side temperature [K].
    Ta : jnp.ndarray
        Outdoor air temperature [K].
    u_hp : jnp.ndarray
        Heat pump modulation signal [-].
    pac : dict
        Heat pump map parameters.

    Returns
    ------
    jnp.ndarray
        Thermal power absorbed at the evaporator [W] (returned with a negative sign).
    """
    T_cn = pac.get("Tcn", PAC_TCN_K)
    T_an = pac.get("Tan", PAC_TAN_K)
    poly = pac["a_e"] + pac["b_e"] * (Tc - T_cn) + pac["c_e"] * (Ta - T_an)
    qe = pac["k_e"] * poly * u_hp
    return -qe


def RC5_state_derivative(state, theta, Ta, Q_solar, Q_con, Q_rad, u_hp):
    """Continuous dynamics of the RC5 model.

    Parameters
    ----------
    state : jnp.ndarray
        Thermal state [Tz, Tw, Ti, Tf, Tc] in K.
    theta : dict
        Thermal + heat pump parameters (sub-dicts 'th' and 'pac').
    Ta : jnp.ndarray
        Outdoor temperature [K].
    Q_solar : jnp.ndarray
        Global incident solar flux [W].
    Q_con : jnp.ndarray
        Convective internal gains [W].
    Q_rad : jnp.ndarray
        Radiative internal gains [W].
    u_hp : jnp.ndarray
        Heat pump modulation signal [-].

    Returns
    ------
    jnp.ndarray
        Time derivative of the state (dTz/dt, dTw/dt, dTi/dt, dTf/dt, dTc/dt).
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
    """Generic RC5 wrapper for `Model_JAX.state_fn`.

    Parameters
    ----------
    x : jnp.ndarray
        Thermal state [Tz, Tw, Ti, Tf, Tc] in K.
    u : Mapping[str, Any]
        Control commands (typically contains 'oveHeaPumY_u').
    d : Mapping[str, Any]
        Disturbances (Ta, Q_solar, internal gains, ...).
    theta : dict
        RC5 model parameters.

    Returns
    ------
    jnp.ndarray
        Continuous state derivative.
    """
    u_hp = jnp.asarray(u["oveHeaPumY_u"])
    Q_con = jnp.asarray(d["InternalGainsCon[1]"])
    Q_rad = jnp.asarray(d["InternalGainsRad[1]"])
    Q_solar = jnp.asarray(d["weaSta_reaWeaHGloHor_y"])
    Ta = jnp.asarray(d["weaSta_reaWeaTDryBul_y"])
    return RC5_state_derivative(jnp.asarray(x), theta, Ta, Q_solar, Q_con, Q_rad, u_hp)


def rc5_qc_dot(x, u, d, theta):
    """Condenser power from state x (wrapper for Qc_dot)."""
    state = jnp.asarray(x)
    return Qc_dot(
        state[-1],  # Tc
        jnp.asarray(d["weaSta_reaWeaTDryBul_y"]),
        jnp.asarray(u["oveHeaPumY_u"]),
        theta["pac"],
    )


def rc5_qe_dot(x, u, d, theta):
    """Evaporator power from state x (wrapper for Qe_dot)."""
    state = jnp.asarray(x)
    return Qe_dot(
        state[-1],  # Tc
        jnp.asarray(d["weaSta_reaWeaTDryBul_y"]),
        jnp.asarray(u["oveHeaPumY_u"]),
        theta["pac"],
    )


def rc5_output_fn(x, u, d, theta):
    """RC5 observation function: returns (Tz, Qc_dot, Qe_dot).

    Parameters
    ----------
    x : jnp.ndarray
        Thermal state [Tz, Tw, Ti, Tf, Tc] in K.
    u : Mapping[str, Any]
        Control commands.
    d : Mapping[str, Any]
        Disturbances.
    theta : dict
        Model parameters.

    Returns
    ------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Zone temperature [K], condenser power [W], evaporator power [W].
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
