from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution

from SIMAX.Models import Model_JAX
from SIMAX.Simulation import Simulation_JAX, SimulationDataset, Sim_and_Data, fit_lm
from SIMAX.Controller import Controller_constSeq
from Identification.Main import THETA_INIT_RC5, TRAIN_CSV, BOUNDS_RC5_S, alpha, GAMMA, CONTROL_COLS, DISTURBANCE_COLS
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
from Identification.Utils import PARAM_ORDER, theta_from_vec
from Identification.Utils import initial_stateRC5




def optimize_theta_initial(popsize=8, maxiter=10, seed=None):
    """Search for an optimized theta_initial via DE, then return theta and cost."""
    dataset = SimulationDataset.from_csv(
        TRAIN_CSV,
        control_cols=CONTROL_COLS,
        disturbance_cols=DISTURBANCE_COLS,
    ).take_fraction(GAMMA)
    dim = len(PARAM_ORDER)
    labels = [f"{section}.{key}" for section, key in PARAM_ORDER]
    out_dir = Path("DE_theta")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_top3_live(hall):
        """Continuously save the top-10 DE candidates as text."""
        if not hall:
            return
        top = hall[:10]
        txt_path = out_dir / "top3_live.txt"
        with txt_path.open("w") as f:
            for rank, (c_lm, c_de, eval_id, vec) in enumerate(top, 1):
                f.write(
                    f"# rank={rank} eval={eval_id} "
                    f"value_lm={c_lm:.6g} cost_de={c_de:.6g}\n"
                )
                for i, label in enumerate(labels):
                    f.write(f"{label} = {float(vec[i]):.6g}\n")
                f.write("\n")

    # best_de: cost used by DE (reweighted)
    # best_lm: pure LM cost (Value = 0.5 * loss_final)
    log_state = {"eval": 0, "best_de": math.inf, "best_lm": math.inf, "hall": [], "archive": {}}
    cb_state = {"nit": 0}

    def objective(vec):
        log_state["eval"] += 1
        eval_idx = log_state["eval"]
        theta = theta_from_vec(jnp.asarray(vec, dtype=jnp.float64), template=THETA_INIT_RC5)
        model = Model_JAX(
            theta=theta,
            state_fn=rc5_state_fn,
            output_fn=rc5_output_fn,
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
        )
        y_meas = jnp.stack(
            (
                dataset.d["reaTZon_y"],
                dataset.d["reaQHeaPumCon_y"],
                dataset.d["reaQHeaPumEva_y"],
            ),
            axis=-1,
        )
        W = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)
        sim_data = Sim_and_Data(simulation=sim, dataset=dataset, y_meas=y_meas, W=W, initial_state_fn=initial_stateRC5)
        fit = fit_lm(
            sim_data,
            bounds=BOUNDS_RC5_S,
            maxiter=50,
            tol=1e-2,
            verbose=False,
        )
        metrics = fit.metrics or {}
        raw_cost = metrics.get("loss_final", math.inf)  # = ||res||^2
        opt_err = metrics.get("opt_error", math.nan)
        iterations = int(metrics.get("iterations", 1))
        status = metrics.get("opt_status", "")

        # Pure LM cost (LM's Value): 0.5 * ||res||^2 if finite, otherwise a penalty.
        if math.isfinite(raw_cost):
            cost_lm = 0.5 * float(raw_cost)
        else:
            cost_lm = 1e6

        # Detect NaN gradient / degenerate optimization
        nan_grad = not math.isfinite(opt_err)
        if nan_grad:
            # Case 1: undefined gradient -> degenerate optimization,
            # apply a strong penalty, slightly reduced if there were many iterations.
            denom = max(iterations, 1)
            base = float(raw_cost) if math.isfinite(raw_cost) else 1e3
            cost_de = (1e3 * base) / float(denom)
        elif not math.isfinite(raw_cost):
            # Case 2: non-finite cost but finite gradient -> numerically unstable trajectory,
            # apply a very large fixed penalty.
            cost_de = 1e6  # large penalty but finite value
        else:
            # "Normal" case: finite cost and finite gradient -> use a reweighting
            # cost_lm * (tol_grad / grad) to encourage still-large gradients,
            # while staying on the same scale as the LM cost (avoids a systematic factor 2).
            base = float(cost_lm)
            tol_grad = float(ident.tol)
            ratio = tol_grad / max(opt_err, tol_grad)
            cost_de = base * max(ratio, 0.5)  # Cap at 50% to avoid over-favoring large gradients vs the cost, which remains objective #1.

        best_de = log_state["best_de"]
        best_lm = log_state["best_lm"]
        it_idx = int((eval_idx - 1) // (popsize * dim))
        best_lm_disp = best_lm if math.isfinite(best_lm) else float("inf")
        vec_arr = jnp.asarray(vec, dtype=jnp.float64)
        vec_np = np.asarray(vec_arr, dtype=float)

        # Update a small global "hall of fame" (top 10, sorted by LM cost).
        hall = log_state.setdefault("hall", [])
        if math.isfinite(cost_lm):
            entry = (float(cost_lm), float(cost_de), int(eval_idx), vec_np)
            hall.append(entry)
            hall.sort(key=lambda t: t[0])
            if len(hall) > 10:
                del hall[10:]
            if entry in hall:
                save_top3_live(hall)
        # Full archive of vectors to recover theta a posteriori
        log_state.setdefault("archive", {})[int(eval_idx)] = vec_np
        named = ", ".join(f"{labels[i]}={float(vec_arr[i]):.4g}" for i in range(dim))

        print("\n================================")
        print(
            f"[DE] eval={log_state['eval']}  iter={it_idx}  "
            f"cost_de={cost_de:.6g}  value_lm={cost_lm:.6g}  best_value={best_lm_disp:.6e}  "
            f"nan_grad={nan_grad}"
        )
        print(f"[DE] theta: {named}")
        print("================================\n")

        # Print the global top 3 right after the current log.
        hall = log_state.get("hall") or []
        if hall:
            print("[DE] Top 3 candidates (global):")
            for rank, (c_lm, c_de, eval_hall, _) in enumerate(hall, 1):
                print(f"   #{rank} eval={eval_hall} value_lm={c_lm:.6g} cost_de={c_de:.6g}")
            print("")

        if cost_de < best_de:
            log_state["best_de"] = cost_de
        if cost_lm < best_lm:
            log_state["best_lm"] = cost_lm

        return cost_de

    def de_callback(xk, convergence):
        """SciPy callback, called once per DE generation. Prints the real nit."""
        cb_state["nit"] += 1
        nit = cb_state["nit"]
        print(
            "================================\n"
            f"[DE-callback] nit={nit}\n"
            "==============================\n"
        )
        return False

    # theta0_vec = np.array([float(THETA_INIT_RC5[s][k]) for s, k in PARAM_ORDER], dtype=float)
    # sigma = 0.2  # gentler perturbation
    # noise = sigma * np.random.randn(popsize, dim)
    # init_pop = theta0_vec * (1.0 + noise)

    # Physical bounds from BOUNDS_RC5, aligned with PARAM_ORDER
    bounds = []
    for section, key in PARAM_ORDER:
        b = BOUNDS_RC5_S[section][key]
        bounds.append((float(b["lb"]), float(b["ub"])))
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    #init_pop = np.clip(init_pop, lb, ub)
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        seed=seed,
        polish=False,
        callback=de_callback,
        #init=init_pop,
        
    )
    print(f"[DE] Done: nit={getattr(result, 'nit', None)}  nfev={getattr(result, 'nfev', None)}")
    hall = log_state.get("hall") or []
    if hall:
        print("\n[DE] Final ranking (Top 3):")
        for rank, (c_lm, c_de, eval_hall, _) in enumerate(hall, 1):
            print(f"   #{rank} eval={eval_hall} value_lm={c_lm:.6g} cost_de={c_de:.6g}")
    best_vec = jnp.asarray(result.x, dtype=jnp.float64)
    best_theta = theta_from_vec(best_vec, template=THETA_INIT_RC5)
    return best_theta, float(result.fun)



__all__ = ["optimize_theta_initial"]

if __name__ == "__main__":
    theta_opt, cost_opt = optimize_theta_initial(popsize=8, maxiter=4, seed=42)
    print("Optimized theta:", theta_opt)
    print("Optimized cost:", cost_opt)
