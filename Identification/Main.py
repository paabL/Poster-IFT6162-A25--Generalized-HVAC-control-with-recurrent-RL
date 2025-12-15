from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path


# Ajout du repo root au PYTHONPATH pour les imports locaux
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
        # Pertes "directes" zone↔extérieur
        # Si tu ne représentes PAS la fenêtre séparément, tu peux prendre R_inf_eff ≈ 0.0066 (infiltration+fenêtre en //).
        "R_inf": jnp.array(0.0115, dtype=jnp.float64),  # K/W  (infiltration seule, ≈0.5 ACH → UA ≈ 87 W/K)  💥

        # Enveloppe via le nœud paroi Tw (mur+toiture) — valeurs qui donnent UA_tot ≈ 125 W/K
        "R_w1": jnp.array(0.00384, dtype=jnp.float64),  # K/W (Tw→Ta)
        "R_w2": jnp.array(0.00414, dtype=jnp.float64),  # K/W (Tz→Tw)  💥

        # Échanges intérieurs
        "R_f":  jnp.array(6.5e-4, dtype=jnp.float64),   # K/W  (plancher↔air zone, h≈8 W/m²K sur 192 m²) 
        "R_i":  jnp.array(3.4e-4, dtype=jnp.float64),   # K/W  (masse interne↔air zone, surfaces larges)  💥
        "R_c":  jnp.array(2.0e-4, dtype=jnp.float64),   # K/W  (eau/coil ↔ dalle)

        # Gains
        "gA":   jnp.array(18.0, dtype=jnp.float64),     # -   (fenêtre S, SHGC~0.7–0.8 → gA ~ 18 m²)

        # Capacités thermiques (J/K)
        "C_z":  jnp.array(6.26e5, dtype=jnp.float64),   # air zone (ρcpV) ≈ 0.626 MJ/K   💥 (x2)
        "C_w":  jnp.array(2.195e7, dtype=jnp.float64),  # murs+toiture ≈ 21.95 MJ/K
        "C_f":  jnp.array(6.306e7, dtype=jnp.float64),  # dalle/plancher ≈ 63.1 MJ/K      💥 (×5)
        "C_c":  jnp.array(5.3e5, dtype=jnp.float64),    # eau boucles ≈ 0.53 MJ/K
        "C_i":  jnp.array(2.57e7, dtype=jnp.float64),   # masse interne ≈ 25.7 MJ/K        💥
    },
    "pac": {
        # Carte simple linéaire autour (Tc_n, Ta_n) = (35°C, 7°C)
        "a_c": jnp.array(14500.0, dtype=jnp.float64),   # W  (≈ puissance nominale à 35/7)
        "b_c": jnp.array(-50.0,   dtype=jnp.float64),   # W/K (sensibilité à Tc, plus modérée)            💥
        "c_c": jnp.array(200.0,   dtype=jnp.float64),   # W/K (sensibilité à Ta)
        "k_c": jnp.array(1.0,     dtype=jnp.float64),
        "a_e": jnp.array(11000.0, dtype=jnp.float64),   # W  (évaporateur dédié)
        "b_e": jnp.array(-40.0,   dtype=jnp.float64),   # W/K
        "c_e": jnp.array(180.0,   dtype=jnp.float64),   # W/K
        "k_e": jnp.array(1.0,     dtype=jnp.float64),
    },
}

BOUNDS_RC5 = {
    "th": {
        # Infiltration seule (pas la fenêtre) — BESTEST ~0.2–1.0 ACH
        # => UA ≈ 35–174 W/K pour ~520 m³ ⇒ R_inf ≈ 0.029–0.005 K/W
        "R_inf": {"lb": 0.6*0.005,  "ub": 0.030},   # 💥

        # Mur+toiture en 2R2C : on vise (R_w1 + R_w2) ≈ 0.004–0.012 K/W (UA ≈ 80–250 W/K)
        "R_w1": {"lb": 0.8*0.0015, "ub": 0.020},   # 💥
        "R_w2": {"lb": 0.8*0.0015, "ub": 0.020},   # 💥

        # Couplages intérieurs (conv+rad ~ 6–10 W/m²K ; A_sol ~192 m² ; masses internes étendues)
        "R_f":  {"lb": 0.2*2e-4,   "ub": 2e-3},    # dalle ↔ air    (UA ≈ 500–5000 W/K)
        "R_i":  {"lb": 1e-4,   "ub": 1.3*2e-3},    # masse ↔ air    (UA ≈ 500–10000 W/K)  💥

        # Solaire fenêtre sud ~24 m², SHGC typ. 0.5–0.8 ⇒ gA ~ 12–19 m²
        "gA":   {"lb": 8.0,    "ub": 2*30.0},

        # Capacités (air ≈ 0.63 MJ/K ; murs/toit ~ 10–60 ; dalle ~ 30–100 ; eau ~ 0.2–1.5 ; masse int. ~ 5–60)
        "C_z":  {"lb": 4e5,    "ub": 1.3*9e5},     # 💥
        "C_w":  {"lb": 0.6*8e6,    "ub": 6e7},
        "C_i":  {"lb": 5e6,    "ub": 1.3*6e7},
        "C_f":  {"lb": 3e7,    "ub": 1.0e8},   # 💥
        "C_c":  {"lb": 0.6*2e5,    "ub": 1.5e6},

        # Échange coil↔dalle (fixe ~ quelques K à ~10–20 kW)
        "R_c":  {"lb": 5e-5,   "ub": 1.2*1e-3},
    },
    "pac": {
        # PAC air-eau ~15 kW @35/7 : on borne à ±40–50 %
        "a_c": {"lb": 8_000.0,  "ub": 22_000.0},   # 💥

        # Sensibilité à Tc : pente modérée (cartes constructeur) 
        "b_c": {"lb": 0.9*-300.0,   "ub": -10.0},      # 💥

        # Sensibilité à Ta : positive, de l’ordre 50–500 W/K
        "c_c": {"lb": 50.0,     "ub": 1.2*500.0},      # 💥

        # Facteur d’échelle — évite de compenser des UA/C mal posés
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
GAMMA = 1.0  # Ratio des données d'entraînement utilisées (0 < gamma ≤ 1)
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
    # Jeu d'entraînement pour l'identification des paramètres, issu de BOPTEST
    dataset = SimulationDataset.from_csv(TRAIN_CSV, control_cols=CONTROL_COLS, disturbance_cols=DISTURBANCE_COLS).take_fraction(GAMMA)

    # État initial à l'équilibre pour RC5
    ta0 = dataset.d["weaSta_reaWeaTDryBul_y"][0]
    qocc0 = dataset.d["InternalGainsCon[1]"][0]
    qocr0 = dataset.d["InternalGainsRad[1]"][0]
    qcd0 = dataset.d["reaQHeaPumCon_y"][0]
    qsol0 = dataset.d["weaSta_reaWeaHGloHor_y"][0]
    x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, THETA_INIT_RC5)

    # Construire un modèle RC5 explicite : dynamique, observation et métadonnées regroupées.
    model = Model_JAX(theta=THETA_INIT_RC5, state_fn=rc5_state_fn, output_fn=rc5_output_fn, state_names=RC5_STATE_NAMES, state_units=RC5_STATE_UNITS, output_names=RC5_OUTPUT_NAMES, output_units=RC5_OUTPUT_UNITS, control_names=RC5_CONTROL_NAMES, control_units=RC5_CONTROL_UNITS, disturbance_names=RC5_DISTURBANCE_NAMES, disturbance_units=RC5_DISTURBANCE_UNITS)
    
    # Simulation JAX sur la grille du dataset avec contrôleur séquence constante.
    simulation = Simulation_JAX(time_grid=dataset.time, d=dataset.d, model=model, controller=Controller_constSeq(oveHeaPumY_u=dataset.u["oveHeaPumY_u"]), integrator="rk2", x0=x0)
    
    # Empilement des mesures dans l'ordre des sorties (Tz, Qc_dot, Qe_dot).
    y_meas = jnp.stack((dataset.d["reaTZon_y"], dataset.d["reaQHeaPumCon_y"], dataset.d["reaQHeaPumEva_y"]), axis=-1)
    
    #Pondérations égales pour les trois sorties.
    W = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)

    # Données de simulation et mesures pour l'identification, le tracé, etc.
    sim_data = Sim_and_Data(simulation=simulation, dataset=dataset, y_meas=y_meas, W=W, initial_state_fn=initial_stateRC5)

    # Identification des paramètres par la méthode des moindres carrés de Levenberg-Marquardt.
    fit = fit_lm(sim_data, bounds=BOUNDS_RC5_S, maxiter=100, tol=1e-3, verbose=True)
    print_report(sim_data, fit, header="Identification summary:")
    sim_data.plot(theta=fit.theta, path=IDENT_PLOT_PATH)

    # Jeu de validation indépendant pour évaluer le modèle identifié.
    val_dataset = SimulationDataset.from_csv(VALIDATION_CSV, control_cols=CONTROL_COLS, disturbance_cols=DISTURBANCE_COLS)
    validator = Validation_RC5(simulation=fit.simulation, dataset=val_dataset)#, initial_state_fn=initial_stateRC5)
    val_state = validator.run(theta=fit.theta)
    val_metrics = validator.report(val_state)
    #print(val_metrics)
    validator.plot(path=VALIDATION_PLOT_PATH)

    # Sauvegarde d'une copie "optimisée" de la simulation pour les scripts externes.
    sim_opti = fit.simulation.copy()
    sim_opti.save_simulation(sim_path)

if __name__ == "__main__":
    main()
