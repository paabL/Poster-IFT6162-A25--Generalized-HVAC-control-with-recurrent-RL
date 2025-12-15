from __future__ import annotations

from pathlib import Path
import sys

# Ajout du repo root au PYTHONPATH pour les imports locaux
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax64  # noqa: F401
import jax.numpy as jnp

from SIMAX.Models import Model_JAX
from SIMAX.Simulation import Simulation_JAX, SimulationDataset, Sim_and_Data, fit_lm, print_report
from SIMAX.Controller import Controller_constSeq
from Identification.Main import THETA_INIT_RC5, BOUNDS_RC5, TRAIN_CSV, VALIDATION_CSV, GAMMA, CONTROL_COLS, DISTURBANCE_COLS, alpha as ALPHA_RC5
from Identification.Models import RC5_STATE_NAMES, RC5_STATE_UNITS, RC5_OUTPUT_NAMES, RC5_OUTPUT_UNITS, RC5_CONTROL_NAMES, RC5_CONTROL_UNITS, RC5_DISTURBANCE_NAMES, RC5_DISTURBANCE_UNITS
from Identification.Utils import RC5_steady_state_sys
from Identification.Validation import Validation_RC5


# Bornes : on identifie uniquement la partie thermique (5 températures),
# la PAC reste figée à ses valeurs initiales.
def _scale_bounds_th(group, alpha):
    def f(v):
        if v["ub"] >= 0:
            lb = v["lb"] / alpha
            ub = v["ub"] * alpha
        else:
            lb = v["lb"] * alpha
            ub = v["ub"] / alpha
        if lb <= ub:
            return {"lb": lb, "ub": ub}
        return {"lb": ub, "ub": lb}

    return {k: f(v) for k, v in group.items()}


BOUNDS_TH_S = _scale_bounds_th(BOUNDS_RC5["th"], 10.0)
THETA_INIT_ZAB = {"th": THETA_INIT_RC5["th"]}
BOUNDS_ZAB = {"th": BOUNDS_TH_S}

FIG_DIR = Path(__file__).parent / "figures"
IDENT_PLOT_PATH_ZAB = FIG_DIR / "identification_main_zab.png"
VALIDATION_PLOT_PATH_ZAB = FIG_DIR / "validation_main_zab.png"
SIM_PATH_ZAB = Path("Models/sim_opti_zab.pkl")


def rc5_state_derivative_meas_qc(state, theta, Ta, Q_solar, Q_con, Q_rad, Qc_meas):
    """Dynamique RC5 en utilisant Qc mesuré (sans carte PAC)."""
    th = theta["th"]
    Tz, Tw, Ti, Tf, Tc = state
    Q_occ = Q_con + Q_rad
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
    dTc = ((Tf - Tc) / th["R_c"] + Qc_meas) / th["C_c"]
    return jnp.array([dTz, dTw, dTi, dTf, dTc])


def rc5_state_fn_zab(x, u, d, theta):
    """Dérivée RC5 pour ZAB, forcée par Qc mesuré."""
    Q_con = jnp.asarray(d["InternalGainsCon[1]"])
    Q_rad = jnp.asarray(d["InternalGainsRad[1]"])
    Q_solar = jnp.asarray(d["weaSta_reaWeaHGloHor_y"])
    Ta = jnp.asarray(d["weaSta_reaWeaTDryBul_y"])
    Qc_meas = jnp.asarray(d["reaQHeaPumCon_y"])
    return rc5_state_derivative_meas_qc(jnp.asarray(x), theta, Ta, Q_solar, Q_con, Q_rad, Qc_meas)


def rc5_output_fn_zab(x, u, d, theta):
    """Observation RC5 pour ZAB : Tz simulé, Qc/Qe mesurés."""
    state = jnp.asarray(x)
    tz = state[0]
    qc = jnp.asarray(d["reaQHeaPumCon_y"])
    qe = jnp.asarray(d["reaQHeaPumEva_y"])
    return tz, qc, qe


def initial_stateRC5_zab(sim_data, theta):
    """État initial RC5 pour ZAB, basé sur Qc mesuré."""
    data = sim_data.dataset
    ta0 = data.d["weaSta_reaWeaTDryBul_y"][0]
    qocc0 = data.d["InternalGainsCon[1]"][0]
    qocr0 = data.d["InternalGainsRad[1]"][0]
    qcd0 = data.d["reaQHeaPumCon_y"][0]
    qsol0 = data.d["weaSta_reaWeaHGloHor_y"][0]
    return RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, theta)


def main():
    # Jeu d'entraînement pour l'identification (mêmes données que Main.py)
    dataset = SimulationDataset.from_csv(
        TRAIN_CSV,
        control_cols=CONTROL_COLS,
        disturbance_cols=DISTURBANCE_COLS,
    ).take_fraction(GAMMA)

    # État initial à l'équilibre pour RC5
    ta0 = dataset.d["weaSta_reaWeaTDryBul_y"][0]
    qocc0 = dataset.d["InternalGainsCon[1]"][0]
    qocr0 = dataset.d["InternalGainsRad[1]"][0]
    qcd0 = dataset.d["reaQHeaPumCon_y"][0]
    qsol0 = dataset.d["weaSta_reaWeaHGloHor_y"][0]
    x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, THETA_INIT_ZAB)

    # Modèle RC5 thermique, forcé par Qc mesuré (la PAC n'est plus modélisée).
    model = Model_JAX(
        theta=THETA_INIT_ZAB,
        state_fn=rc5_state_fn_zab,
        output_fn=rc5_output_fn_zab,
        state_names=RC5_STATE_NAMES,
        state_units=RC5_STATE_UNITS,
        output_names=RC5_OUTPUT_NAMES,
        output_units=RC5_OUTPUT_UNITS,
        control_names=RC5_CONTROL_NAMES,
        control_units=RC5_CONTROL_UNITS,
        disturbance_names=RC5_DISTURBANCE_NAMES,
        disturbance_units=RC5_DISTURBANCE_UNITS,
    )

    simulation = Simulation_JAX(
        time_grid=dataset.time,
        d=dataset.d,
        model=model,
        controller=Controller_constSeq(oveHeaPumY_u=dataset.u["oveHeaPumY_u"]),
        integrator="rk2",
        x0=x0,
    )

    # Mesures dans l'ordre des sorties (Tz, Qc_dot, Qe_dot)
    y_meas = jnp.stack(
        (
            dataset.d["reaTZon_y"],
            dataset.d["reaQHeaPumCon_y"],
            dataset.d["reaQHeaPumEva_y"],
        ),
        axis=-1,
    )

    # Critère d'identification : uniquement Tz (poids nuls sur Qc_dot, Qe_dot).
    W = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

    sim_data = Sim_and_Data(
        simulation=simulation,
        dataset=dataset,
        y_meas=y_meas,
        W=W,
        initial_state_fn=initial_stateRC5_zab,
    )

    fit = fit_lm(
        sim_data,
        bounds=BOUNDS_ZAB,
        maxiter=100,
        tol=1e-6,
        verbose=True,
    )
    print_report(sim_data, fit, header="Identification (RC5 thermique seul)")
    sim_data.plot(theta=fit.theta, path=IDENT_PLOT_PATH_ZAB)

    # Validation sur dataset indépendant, avec état initial cohérent avec le dataset de validation.
    val_dataset = SimulationDataset.from_csv(
        VALIDATION_CSV,
        control_cols=CONTROL_COLS,
        disturbance_cols=DISTURBANCE_COLS,
    )
    ta0_v = val_dataset.d["weaSta_reaWeaTDryBul_y"][0]
    qocc0_v = val_dataset.d["InternalGainsCon[1]"][0]
    qocr0_v = val_dataset.d["InternalGainsRad[1]"][0]
    qcd0_v = val_dataset.d["reaQHeaPumCon_y"][0]
    qsol0_v = val_dataset.d["weaSta_reaWeaHGloHor_y"][0]
    x0_v = RC5_steady_state_sys(ta0_v, qsol0_v, qocc0_v, qocr0_v, qcd0_v, fit.theta)
    sim_for_val = fit.simulation.copy(x0=x0_v)
    validator = Validation_RC5(simulation=sim_for_val, dataset=val_dataset)
    val_state = validator.run(theta=fit.theta)
    _ = validator.report(val_state)
    validator.plot(path=VALIDATION_PLOT_PATH_ZAB)

    # Sauvegarde d'une simulation optimisée spécifique à cette identification.
    sim_opti = fit.simulation.copy()
    sim_opti.save_simulation(SIM_PATH_ZAB)


if __name__ == "__main__":
    main()
