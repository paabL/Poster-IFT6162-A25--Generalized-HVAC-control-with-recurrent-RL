from __future__ import annotations

import math
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution

# Ajout du repo root au PYTHONPATH pour les imports locaux
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SIMAX.Models import Model_JAX
from SIMAX.Simulation import Simulation_JAX, SimulationDataset, Sim_and_Data, fit_lm
from SIMAX.Controller import Controller_constSeq
from Identification.Main import TRAIN_CSV, GAMMA, CONTROL_COLS, DISTURBANCE_COLS
from Identification.Main_zab import THETA_INIT_ZAB, BOUNDS_ZAB, rc5_state_fn_zab, rc5_output_fn_zab, initial_stateRC5_zab
from Identification.Models import (
    RC5_STATE_NAMES,
    RC5_STATE_UNITS,
    RC5_OUTPUT_NAMES,
    RC5_OUTPUT_UNITS,
    RC5_CONTROL_NAMES,
    RC5_CONTROL_UNITS,
    RC5_DISTURBANCE_NAMES,
    RC5_DISTURBANCE_UNITS,
)


def optimize_theta_initial_zab(popsize=8, maxiter=10, seed=None):
    """Recherche un theta_initial thermique optimisé via DE (ZAB)."""
    dataset = SimulationDataset.from_csv(
        TRAIN_CSV,
        control_cols=CONTROL_COLS,
        disturbance_cols=DISTURBANCE_COLS,
    ).take_fraction(GAMMA)

    th_keys = list(THETA_INIT_ZAB["th"].keys())
    dim = len(th_keys)
    labels = [f"th.{key}" for key in th_keys]
    out_dir = Path("DE_theta_zab")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_top10_live(hall):
        """Sauvegarde en continu le top-10 DE en texte."""
        if not hall:
            return
        top = hall[:10]
        txt_path = out_dir / "top10_live.txt"
        with txt_path.open("w") as f:
            for rank, (c_lm, c_de, eval_id, vec) in enumerate(top, 1):
                f.write(
                    f"# rank={rank} eval={eval_id} "
                    f"value_lm={c_lm:.6g} cost_de={c_de:.6g}\n"
                )
                for i, label in enumerate(labels):
                    f.write(f"{label} = {float(vec[i]):.6g}\n")
                f.write("\n")

    def vec_to_theta(vec):
        """Vecteur -> dictionnaire theta ZAB (th seulement)."""
        th = {}
        for i, key in enumerate(th_keys):
            th[key] = jnp.asarray(vec[i], dtype=jnp.float64)
        return {"th": th}

    # best_de : coût utilisé par DE (repondéré)
    # best_lm : coût LM pur (Value = 0.5 * loss_final)
    log_state = {"eval": 0, "best_de": math.inf, "best_lm": math.inf, "hall": [], "archive": {}}
    cb_state = {"nit": 0}

    def objective(vec):
        log_state["eval"] += 1
        eval_idx = log_state["eval"]
        vec_arr = jnp.asarray(vec, dtype=jnp.float64)
        theta = vec_to_theta(vec_arr)

        model = Model_JAX(
            theta=theta,
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
        sim = Simulation_JAX(
            time_grid=dataset.time,
            d=dataset.d,
            model=model,
            controller=Controller_constSeq(oveHeaPumY_u=dataset.u["oveHeaPumY_u"]),
            integrator="rk2",
            x0=jnp.full((5,), 293.15, dtype=jnp.float64),
        )
        y_meas = jnp.stack(
            (
                dataset.d["reaTZon_y"],
                dataset.d["reaQHeaPumCon_y"],
                dataset.d["reaQHeaPumEva_y"],
            ),
            axis=-1,
        )
        W = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
        sim_data = Sim_and_Data(simulation=sim, dataset=dataset, y_meas=y_meas, W=W, initial_state_fn=initial_stateRC5_zab)
        fit = fit_lm(
            sim_data,
            bounds=BOUNDS_ZAB,
            maxiter=50,
            tol=1e-2,
            verbose=False,
        )
        metrics = fit.metrics or {}
        raw_cost = metrics.get("loss_final", math.inf)  # = ||res||^2
        opt_err = metrics.get("opt_error", math.nan)
        iterations = int(metrics.get("iterations", 1))

        # Coût LM pur (Value du LM) : 0.5 * ||res||^2 si fini, sinon pénalité.
        if math.isfinite(raw_cost):
            cost_lm = 0.5 * float(raw_cost)
        else:
            cost_lm = 1e6

        # Détection d'un gradient NaN / optimisation dégénérée
        nan_grad = not math.isfinite(opt_err)
        if nan_grad:
            denom = max(iterations, 1)
            base = float(raw_cost) if math.isfinite(raw_cost) else 1e3
            cost_de = (1e3 * base) / float(denom)
        elif not math.isfinite(raw_cost):
            cost_de = 1e6
        else:
            base = float(cost_lm)
            tol_grad = 1e-3
            ratio = tol_grad / max(opt_err, tol_grad)
            cost_de = base * max(ratio, 0.5)

        best_de = log_state["best_de"]
        best_lm = log_state["best_lm"]
        it_idx = int((eval_idx - 1) // (popsize * dim))
        best_lm_disp = best_lm if math.isfinite(best_lm) else float("inf")
        vec_np = np.asarray(vec_arr, dtype=float)

        hall = log_state.setdefault("hall", [])
        if math.isfinite(cost_lm):
            entry = (float(cost_lm), float(cost_de), int(eval_idx), vec_np)
            hall.append(entry)
            hall.sort(key=lambda t: t[0])
            if len(hall) > 10:
                del hall[10:]
            if entry in hall:
                save_top10_live(hall)
        log_state.setdefault("archive", {})[int(eval_idx)] = vec_np
        named = ", ".join(f"{labels[i]}={float(vec_arr[i]):.6g}" for i in range(dim))

        print("\n================================")
        print(
            f"[DE-ZAB] eval={log_state['eval']}  iter={it_idx}  "
            f"cost_de={cost_de:.6g}  value_lm={cost_lm:.6g}  best_value={best_lm_disp:.6e}  "
            f"nan_grad={nan_grad}"
        )
        print(f"[DE-ZAB] theta_th: {named}")
        print("================================\n")

        hall = log_state.get("hall") or []
        if hall:
            print("[DE-ZAB] Top 3 pistes (global) :")
            for rank, (c_lm, c_de, eval_hall, _) in enumerate(hall, 1):
                print(f"   #{rank} eval={eval_hall} value_lm={c_lm:.6g} cost_de={c_de:.6g}")
            print("")

        if cost_de < best_de:
            log_state["best_de"] = cost_de
        if cost_lm < best_lm:
            log_state["best_lm"] = cost_lm

        return cost_de

    def de_callback(xk, convergence):
        """Callback SciPy, appelé une fois par génération DE."""
        cb_state["nit"] += 1
        nit = cb_state["nit"]
        print(
            "================================\n"
            f"[DE-ZAB-callback] nit={nit}\n"
            "==============================\n"
        )
        return False

    # Bornes physiques issues de BOUNDS_ZAB, alignées sur l'ordre des clés th.
    bounds = []
    for key in th_keys:
        b = BOUNDS_ZAB["th"][key]
        bounds.append((float(b["lb"]), float(b["ub"])))

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        seed=seed,
        polish=False,
        callback=de_callback,
    )
    print(f"[DE-ZAB] Terminé: nit={getattr(result, 'nit', None)}  nfev={getattr(result, 'nfev', None)}")

    hall = log_state.get("hall") or []
    if hall:
        print("\n[DE-ZAB] Classement final (Top 3) :")
        for rank, (c_lm, c_de, eval_hall, _) in enumerate(hall, 1):
            print(f"   #{rank} eval={eval_hall} value_lm={c_lm:.6g} cost_de={c_de:.6g}")

    best_vec = jnp.asarray(result.x, dtype=jnp.float64)
    best_theta = vec_to_theta(best_vec)
    return best_theta, float(result.fun)


__all__ = ["optimize_theta_initial_zab"]


if __name__ == "__main__":
    theta_opt, cost_opt = optimize_theta_initial_zab(popsize=8, maxiter=4, seed=42)
    print("Optimized theta (ZAB):", theta_opt)
    print("Optimized cost (ZAB):", cost_opt)
