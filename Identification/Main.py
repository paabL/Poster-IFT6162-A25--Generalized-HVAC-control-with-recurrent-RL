from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path


# Add repo root to PYTHONPATH for local imports
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import jax64  # noqa: F401
import jax.numpy as jnp

from SIMAX.Models import Model_JAX
from SIMAX.Simulation import Simulation_JAX, SimulationDataset, Sim_and_Data, fit_lm, print_report
from SIMAX.Controller import Controller_constSeq, Controller_Constant, Controller_Proportional
from Identification.Models import (
    rc5_state_fn,
    rc5_output_fn,
    RC5_STATE_NAMES,
    RC5_STATE_UNITS,
    RC5_OUTPUT_NAMES,
    RC5_OUTPUT_UNITS,
    RC5_CONTROL_NAMES,
    RC5_CONTROL_UNITS,
    RC5_DISTURBANCE_NAMES,
    RC5_DISTURBANCE_UNITS,
)
from Identification.Utils import initial_stateRC5, RC5_steady_state_sys
from Identification.Validation import Validation_RC5

THETA_INIT_RC5 = {
    "th": {
        # "Direct" zone↔outdoor losses
        # If you do NOT model the window separately, you can take R_inf_eff ≈ 0.0066 (infiltration+window in parallel).
        "R_inf": jnp.array(0.0115, dtype=jnp.float64),  # K/W  (infiltration only, ≈0.5 ACH → UA ≈ 87 W/K)  💥

        # Envelope via the wall node Tw (wall+roof) — values that give UA_tot ≈ 125 W/K
        "R_w1": jnp.array(0.00384, dtype=jnp.float64),  # K/W (Tw→Ta)
        "R_w2": jnp.array(0.00414, dtype=jnp.float64),  # K/W (Tz→Tw)  💥

        # Indoor exchanges
        "R_f":  jnp.array(6.5e-4, dtype=jnp.float64),   # K/W  (floor↔zone air, h≈8 W/m²K over 192 m²)
        "R_i":  jnp.array(3.4e-4, dtype=jnp.float64),   # K/W  (internal mass↔zone air, large surfaces)  💥
        "R_c":  jnp.array(2.0e-4, dtype=jnp.float64),   # K/W  (water/coil ↔ slab)

        # Gains
        "gA":   jnp.array(18.0, dtype=jnp.float64),     # -   (south window, SHGC~0.7–0.8 → gA ~ 18 m²)

        # Thermal capacities (J/K)
        "C_z":  jnp.array(6.26e5, dtype=jnp.float64),   # air zone (ρcpV) ≈ 0.626 MJ/K   💥 (x2)
        "C_w":  jnp.array(2.195e7, dtype=jnp.float64),  # walls+roof ≈ 21.95 MJ/K
        "C_f":  jnp.array(6.306e7, dtype=jnp.float64),  # slab/floor ≈ 63.1 MJ/K      💥 (×5)
        "C_c":  jnp.array(5.3e5, dtype=jnp.float64),    # loop water ≈ 0.53 MJ/K
        "C_i":  jnp.array(2.57e7, dtype=jnp.float64),   # internal mass ≈ 25.7 MJ/K        💥
    },
    "pac": {
        # Simple linear map around (Tc_n, Ta_n) = (35°C, 7°C)
        "a_c": jnp.array(14500.0, dtype=jnp.float64),   # W  (≈ nominal power at 35/7)
        "b_c": jnp.array(-50.0,   dtype=jnp.float64),   # W/K (Tc sensitivity, more moderate)            💥
        "c_c": jnp.array(200.0,   dtype=jnp.float64),   # W/K (Ta sensitivity)
        "k_c": jnp.array(1.0,     dtype=jnp.float64),
        "a_e": jnp.array(11000.0, dtype=jnp.float64),   # W  (dedicated evaporator)
        "b_e": jnp.array(-40.0,   dtype=jnp.float64),   # W/K
        "c_e": jnp.array(180.0,   dtype=jnp.float64),   # W/K
        "k_e": jnp.array(1.0,     dtype=jnp.float64),
    },
}

BOUNDS_RC5 = {
    "th": {
        # Infiltration only (not the window) — BESTEST ~0.2–1.0 ACH
        # => UA ≈ 35–174 W/K for ~520 m³ ⇒ R_inf ≈ 0.029–0.005 K/W
        "R_inf": {"lb": 0.6*0.005,  "ub": 0.030},   # 💥

        # Wall+roof as 2R2C: target (R_w1 + R_w2) ≈ 0.004–0.012 K/W (UA ≈ 80–250 W/K)
        "R_w1": {"lb": 0.8*0.0015, "ub": 0.020},   # 💥
        "R_w2": {"lb": 0.8*0.0015, "ub": 0.020},   # 💥

        # Indoor couplings (conv+rad ~ 6–10 W/m²K; A_floor ~192 m²; large internal masses)
        "R_f":  {"lb": 0.2*2e-4,   "ub": 2e-3},    # slab ↔ air (UA ≈ 500–5000 W/K)
        "R_i":  {"lb": 1e-4,   "ub": 1.3*2e-3},    # mass ↔ air (UA ≈ 500–10000 W/K)  💥

        # South window solar ~24 m², typical SHGC 0.5–0.8 ⇒ gA ~ 12–19 m²
        "gA":   {"lb": 8.0,    "ub": 2*30.0},

        # Capacities (air ≈ 0.63 MJ/K; walls/roof ~ 10–60; slab ~ 30–100; water ~ 0.2–1.5; internal mass ~ 5–60)
        "C_z":  {"lb": 4e5,    "ub": 1.3*9e5},     # 💥
        "C_w":  {"lb": 0.6*8e6,    "ub": 6e7},
        "C_i":  {"lb": 5e6,    "ub": 1.3*6e7},
        "C_f":  {"lb": 3e7,    "ub": 1.0e8},   # 💥
        "C_c":  {"lb": 0.6*2e5,    "ub": 1.5e6},

        # Coil↔slab exchange (fixed ~ a few K at ~10–20 kW)
        "R_c":  {"lb": 5e-5,   "ub": 1.2*1e-3},
    },
    "pac": {
        # Air-to-water heat pump ~15 kW @35/7: bound to ±40–50%
        "a_c": {"lb": 8_000.0,  "ub": 22_000.0},   # 💥

        # Tc sensitivity: moderate slope (manufacturer maps)
        "b_c": {"lb": 0.9*-300.0,   "ub": -10.0},      # 💥

        # Ta sensitivity: positive, on the order of 50–500 W/K
        "c_c": {"lb": 50.0,     "ub": 1.2*500.0},      # 💥

        # Scaling factor — avoids compensating poorly set UA/C
        "k_c": {"lb": 0.6,      "ub": 1.2*1.2},
        "a_e": {"lb": 6_000.0,  "ub": 20_000.0},
        "b_e": {"lb": -300.0,   "ub": -5.0},
        "c_e": {"lb": 50.0,     "ub": 1.2*500.0},
        "k_e": {"lb": 0.6,      "ub": 1.2*1.2},
    },
}

alpha = 2.0

def scaled_bounds(alpha):
    f = lambda v: (lambda x, y: {"lb": x, "ub": y} if x < y else {"lb": y, "ub": x})(
        *((v["lb"]/alpha, v["ub"]*alpha) if v["ub"] >= 0 else (v["lb"]*alpha, v["ub"]/alpha))
    )
    return {g: {k: f(v) for k, v in G.items()} for g, G in BOUNDS_RC5.items()}

BOUNDS_RC5_S = scaled_bounds(alpha)

TRAIN_CSV = "datas/train_df.csv"
VALIDATION_CSV = "datas/validation_df.csv"
IDENT_PLOT_PATH = "figures/identification12_main.png"
VALIDATION_PLOT_PATH = "figures/validation_main.png"
GAMMA = 1.0  # Fraction of training data used (0 < gamma ≤ 1)
sim_path = Path("Models/sim_opti.pkl")

CONTROL_COLS = ("oveHeaPumY_u",)
DISTURBANCE_COLS = (
    "InternalGainsCon[1]",
    "InternalGainsRad[1]",
    "weaSta_reaWeaHGloHor_y",
    "weaSta_reaWeaTDryBul_y",
    "reaTZon_y",
    "reaQHeaPumCon_y",
    "reaQHeaPumEva_y",
)

def main():
    # Training set for parameter identification, from BOPTEST
    dataset = SimulationDataset.from_csv(TRAIN_CSV, control_cols=CONTROL_COLS, disturbance_cols=DISTURBANCE_COLS).take_fraction(GAMMA)

    # Steady-state initial state for RC5
    ta0 = dataset.d["weaSta_reaWeaTDryBul_y"][0]
    qocc0 = dataset.d["InternalGainsCon[1]"][0]
    qocr0 = dataset.d["InternalGainsRad[1]"][0]
    qcd0 = dataset.d["reaQHeaPumCon_y"][0]
    qsol0 = dataset.d["weaSta_reaWeaHGloHor_y"][0]
    x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, THETA_INIT_RC5)

    # Build an explicit RC5 model: dynamics, observation and metadata bundled together.
    model = Model_JAX(theta=THETA_INIT_RC5, state_fn=rc5_state_fn, output_fn=rc5_output_fn, state_names=RC5_STATE_NAMES, state_units=RC5_STATE_UNITS, output_names=RC5_OUTPUT_NAMES, output_units=RC5_OUTPUT_UNITS, control_names=RC5_CONTROL_NAMES, control_units=RC5_CONTROL_UNITS, disturbance_names=RC5_DISTURBANCE_NAMES, disturbance_units=RC5_DISTURBANCE_UNITS)
    
    # JAX simulation on the dataset grid with a constant-sequence controller.
    simulation = Simulation_JAX(time_grid=dataset.time, d=dataset.d, model=model, controller=Controller_constSeq(oveHeaPumY_u=dataset.u["oveHeaPumY_u"]), integrator="rk2", x0=x0)
    
    # Stack measurements in the output order (Tz, Qc_dot, Qe_dot).
    y_meas = jnp.stack((dataset.d["reaTZon_y"], dataset.d["reaQHeaPumCon_y"], dataset.d["reaQHeaPumEva_y"]), axis=-1)
    
    # Equal weights for the three outputs.
    W = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)

    # Simulation/measurement coupling for identification, plotting, etc.
    sim_data = Sim_and_Data(simulation=simulation, dataset=dataset, y_meas=y_meas, W=W, initial_state_fn=initial_stateRC5)

    # Parameter identification via Levenberg–Marquardt least squares.
    fit = fit_lm(sim_data, bounds=BOUNDS_RC5_S, maxiter=100, tol=1e-3, verbose=True)
    print_report(sim_data, fit, header="Identification summary:")
    sim_data.plot(theta=fit.theta, path=IDENT_PLOT_PATH)

    # Independent validation set to assess the identified model.
    val_dataset = SimulationDataset.from_csv(VALIDATION_CSV, control_cols=CONTROL_COLS, disturbance_cols=DISTURBANCE_COLS)
    validator = Validation_RC5(simulation=fit.simulation, dataset=val_dataset)#, initial_state_fn=initial_stateRC5)
    val_state = validator.run(theta=fit.theta)
    val_metrics = validator.report(val_state)
    #print(val_metrics)
    validator.plot(path=VALIDATION_PLOT_PATH)

    # Save an "optimized" copy of the simulation for external scripts.
    sim_opti = fit.simulation.copy()
    sim_opti.save_simulation(sim_path)

if __name__ == "__main__":
    main()
