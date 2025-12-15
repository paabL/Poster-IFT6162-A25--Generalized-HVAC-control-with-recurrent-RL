from __future__ import annotations

import sys
from pathlib import Path

# Allows running this script from `MPC/` (project-relative imports).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax64  # noqa: F401
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from Utils.Occup import occupancy_probability
from SIMAX.Controller import Controller_MPC, Controller_PID, Controller_constSeq
from Utils.utils import RC5_steady_state_sys, scale_rc5_building
from Utils.rc5_cost import interval_components
from gymRC5 import dataset_short, sim_opti_loaded

# -------------------- Config (aligned with test_1FALSTM) --------------------
START_TIME_S = 28 * 24 * 3600  # Feb 1st if t=0 = Jan 1st
WARMUP_DAYS = 4
EPISODE_DAYS = 7

BASE_SETPOINT_K = 273.15 + 21.0  # warmup PID
MPC_STEP_S = 300                # recompute u every hour
MPC_HORIZON_H = 24               # horizon MPC
INTEGRATOR = "euler"

# cost = W[0]*energy(€) + W[1]*comfort(K·h)
# + W[2]*∫u² dt (unit²·h) to regularize the control.
W = jnp.asarray([1.0, 5.0, 0.05], dtype=jnp.float64)
OUT_PATH = ROOT / "MPC" / "figures" / "mpc_rc5_week_feb_warmup4.png"

# k: changes building properties (scaling of thermal parameters "th")
K: dict[str, float] = {"k_size": 1.1, "k_U": 0.9, "k_inf": 1.1, "k_win": 0.9, "k_mass": 1.1}


def _window(ds, t0_s: float, t1_s: float) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    t = np.asarray(ds.time, dtype=float)
    i0 = int(np.searchsorted(t, float(t0_s), side="left"))
    i1 = int(np.searchsorted(t, float(t1_s), side="left"))
    if i0 < 0 or i1 >= t.size:
        raise ValueError("Warmup/episode window out of the dataset.")
    sl = slice(i0, i1 + 1)
    return ds.time[sl], {k: v[sl] for k, v in ds.d.items()}


def cost_core(
    u_window_bloc,
    x_i,
    i,
    setpoints,  # unused (comfort bands used)
    sim,
    time_grid,  # unused
    window_size,
    n,
    forecast=None,  # unused
):
    i = int(i)
    window_size = int(window_size)
    n = int(n)

    end_idx = min(i + window_size, int(sim.time_grid.shape[0]))
    horizon_len = end_idx - i
    if horizon_len < 2:
        return jnp.zeros((3,), dtype=jnp.float64)

    window_grid = sim.time_grid[i:end_idx]
    u_window = jnp.clip(jnp.repeat(u_window_bloc, n)[:horizon_len], 0.0, 1.0)

    controller = Controller_constSeq(oveHeaPumY_u=u_window)
    t, y_sim, _states, _controls = sim.run(
        time_grid=window_grid,
        x0=jnp.asarray(x_i, dtype=jnp.float64),
        controller=controller,
    )

    y = jnp.asarray(y_sim, dtype=jnp.float64)
    if y.ndim == 1:
        y = y[:, None]
    tz = y[:, 0]
    qc = y[:, 1] if y.shape[1] > 1 else jnp.zeros_like(tz)
    qe = y[:, 2] if y.shape[1] > 2 else jnp.zeros_like(tz)

    lower = jnp.asarray(sim.d["LowerSetp[1]"][i:end_idx], dtype=jnp.float64)
    upper = jnp.asarray(sim.d["UpperSetp[1]"][i:end_idx], dtype=jnp.float64)
    occ = jnp.asarray(sim.d["occupancy"][i:end_idx], dtype=jnp.float64)
    price = jnp.asarray(sim.d["electricity_price"][i:end_idx], dtype=jnp.float64)

    # Alignment "like gymRC5": costs on intervals (t[:-1])
    t_step = t[:-1]  # seconds, N
    tz_seq = tz[1:]  # K, N
    php = (qc - jnp.abs(qe))[:-1]  # W, N

    energy_cost, comfort_cost, _u_unit_h, _tz_kh, _sat_unit_h = interval_components(
        t_step_s=t_step,
        tz_seq_k=tz_seq,
        lower_seq_k=lower[:-1],
        upper_seq_k=upper[:-1],
        occ_seq=occ[:-1],
        php_w_seq=php,
        price_seq=price[:-1],
        comfort_huber_k=0.2,
        delta_sat_seq=None,
        xp=jnp,
    )

    u2_cost = jnp.trapezoid(jnp.square(u_window[:-1]), x=t_step) / 3600.0
    return jnp.asarray([energy_cost, comfort_cost, u2_cost], dtype=jnp.float64)


def _plot(
    *,
    warmup: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, jnp.ndarray]],
    episode: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, jnp.ndarray], jnp.ndarray],
    forecasts: list[dict] | None,
    nZOH: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    t_w, y_w, u_w, d_w = warmup
    t, y, u, d, setpoints = episode
    nZOH = int(max(1, nZOH))

    def _days(x) -> np.ndarray:
        return np.asarray(x, dtype=float) / 86400.0

    # "Main" data (episode) at dataset resolution
    t_days_main = _days(t)
    tz_c = np.asarray(y[:, 0], dtype=float) - 273.15

    u_arr_raw = np.asarray(u.get("oveHeaPumY_u", np.zeros_like(tz_c)), dtype=float)
    if u_arr_raw.size == t_days_main.size:
        u_arr = u_arr_raw
    elif u_arr_raw.size == max(0, t_days_main.size - 1):
        u_arr = np.concatenate([u_arr_raw, u_arr_raw[-1:]]) if u_arr_raw.size else np.zeros_like(tz_c)
    else:
        # fallback: best-effort match to the length of t (plot only)
        u_arr = np.resize(u_arr_raw, t_days_main.size) if u_arr_raw.size else np.zeros_like(tz_c)

    qc_arr = np.asarray(y[:, 1], dtype=float) if y.shape[1] > 1 else np.zeros_like(tz_c)
    qe_arr = np.asarray(y[:, 2], dtype=float) if y.shape[1] > 2 else np.zeros_like(tz_c)
    php_arr = qc_arr - np.abs(qe_arr)

    # Consumption (kWh) over the episode from P_hp (W)
    if t_days_main.size >= 2:
        energy_kwh = float(np.trapezoid(np.maximum(php_arr, 0.0) / 1000.0, x=t_days_main * 24.0))
    else:
        energy_kwh = 0.0

    lower_c = np.asarray(d["LowerSetp[1]"], dtype=float) - 273.15
    upper_c = np.asarray(d["UpperSetp[1]"], dtype=float) - 273.15
    ta = np.asarray(d["weaSta_reaWeaTDryBul_y"], dtype=float) - 273.15
    qsol = np.asarray(d["weaSta_reaWeaHGloHor_y"], dtype=float)
    qocc = np.asarray(d["InternalGainsCon[1]"], dtype=float)
    qocr = np.asarray(d["InternalGainsRad[1]"], dtype=float)
    occ = np.asarray(d["occupancy"], dtype=float)
    prob = occupancy_probability(np.asarray(t, dtype=float))
    price = np.asarray(d["electricity_price"], dtype=float)

    # "RL step" ≈ MPC step (ZOH)
    t_np = np.asarray(t, dtype=float)
    t_left = t_np[:-1]  # seconds
    blocks = np.arange(0, t_left.size, nZOH, dtype=int)
    t_days_rl = _days(t_np[blocks]) if blocks.size else np.asarray([], dtype=float)
    sp_rl = (np.asarray(setpoints, dtype=float) - 273.15)[blocks] if blocks.size else np.asarray([], dtype=float)

    # Reward/terms per MPC step (same conventions as gymRC5: negative terms, in euros)
    tz_k = np.asarray(y[:, 0], dtype=float)
    lower_k = np.asarray(d["LowerSetp[1]"], dtype=float)
    upper_k = np.asarray(d["UpperSetp[1]"], dtype=float)
    occ_k = np.asarray(d["occupancy"], dtype=float)
    price_k = np.asarray(d["electricity_price"], dtype=float)
    php_watts = np.asarray(php_arr, dtype=float)

    tz_seq_k = tz_k[1:]  # interval-aligned
    comfort_dev_k = np.maximum(lower_k[:-1] - tz_seq_k, 0.0) + np.maximum(tz_seq_k - upper_k[:-1], 0.0)
    comfort_y = comfort_dev_k * occ_k[:-1]  # K
    energy_y = price_k[:-1] * (np.maximum(php_watts[:-1], 0.0) / 1000.0)  # €/h

    u_interval = u_arr[:-1] if u_arr.size >= 2 else np.asarray([], dtype=float)
    u2_y = np.square(u_interval[: t_left.size]) if u_interval.size else np.zeros_like(t_left)

    w_energy = float(np.asarray(W[0]))
    w_comfort = float(np.asarray(W[1]))
    w_sat = float(np.asarray(W[2]))

    energy_terms = []
    comfort_terms = []
    sat_terms = []
    rewards = []
    for s in blocks:
        e = min(int(s + nZOH), int(t_left.size))
        if (e - s) >= 2:
            energy_eur = float(np.trapezoid(energy_y[s:e], x=t_left[s:e] / 3600.0))
            comfort_kh = float(np.trapezoid(comfort_y[s:e], x=t_left[s:e]) / 3600.0)
            sat_unit2_h = float(np.trapezoid(u2_y[s:e], x=t_left[s:e]) / 3600.0)
        else:
            energy_eur = 0.0
            comfort_kh = 0.0
            sat_unit2_h = 0.0

        energy_term = -w_energy * energy_eur
        comfort_term = -w_comfort * comfort_kh
        sat_term = -w_sat * sat_unit2_h
        reward = comfort_term + energy_term + sat_term

        energy_terms.append(energy_term)
        comfort_terms.append(comfort_term)
        sat_terms.append(sat_term)
        rewards.append(reward)

    rewards = np.asarray(rewards, dtype=float)
    indiv = np.stack([np.asarray(comfort_terms), np.asarray(energy_terms), np.asarray(sat_terms)], axis=1) if rewards.size else None

    reward_label = "reward"
    comfort_label = "comfort"
    energy_label = "energy"
    sat_label = "saturation"
    if indiv is not None and indiv.shape[1] >= 3:
        total_reward = float(rewards.sum())
        total_reward_safe = total_reward if abs(total_reward) > 1e-6 else 1.0
        total_comfort = float(indiv[:, 0].sum())
        total_energy = float(indiv[:, 1].sum())
        total_sat = float(indiv[:, 2].sum())
        comfort_pct = 100.0 * total_comfort / total_reward_safe
        energy_pct = 100.0 * total_energy / total_reward_safe
        sat_pct = 100.0 * total_sat / total_reward_safe
        reward_label = f"reward (sum={total_reward:.2f}€)"
        comfort_label = f"comfort ({comfort_pct:.0f}%, {total_comfort:.2f}€)"
        energy_label = f"energy ({energy_pct:.0f}%, {total_energy:.2f}€)"
        sat_label = f"sat ({sat_pct:.0f}%, {total_sat:.2f}€)"

    # Warmup (format and style aligned with gymRC5.MyMinimalEnv._plot_episode)
    warm_time = _days(t_w)
    warm_tz = (np.asarray(y_w[:, 0], dtype=float) - 273.15) if y_w.size else np.asarray([], dtype=float)
    warm_u_raw = np.asarray(u_w.get("oveHeaPumY_u", np.array([], dtype=float)), dtype=float)
    warm_u = warm_u_raw
    warm_qc = np.asarray(y_w[:, 1], dtype=float) if y_w.shape[1] > 1 else np.zeros_like(warm_tz)
    warm_qe = np.asarray(y_w[:, 2], dtype=float) if y_w.shape[1] > 2 else np.zeros_like(warm_tz)
    warm_php = warm_qc - np.abs(warm_qe)

    has_warmup = warm_time.size > 0
    warmup_span = (float(warm_time[0]), float(warm_time[-1])) if has_warmup else None
    if has_warmup:
        w_lower_c = np.asarray(d_w["LowerSetp[1]"], dtype=float) - 273.15
        w_upper_c = np.asarray(d_w["UpperSetp[1]"], dtype=float) - 273.15
        w_ta = np.asarray(d_w["weaSta_reaWeaTDryBul_y"], dtype=float) - 273.15
        w_qsol = np.asarray(d_w["weaSta_reaWeaHGloHor_y"], dtype=float)
        w_qocc = np.asarray(d_w["InternalGainsCon[1]"], dtype=float)
        w_qocr = np.asarray(d_w["InternalGainsRad[1]"], dtype=float)
        w_occ = np.asarray(d_w["occupancy"], dtype=float)
        w_price = np.asarray(d_w["electricity_price"], dtype=float)
        w_sp = np.full_like(warm_time, BASE_SETPOINT_K - 273.15, dtype=float)
        w_prob = occupancy_probability(np.asarray(t_w, dtype=float))

    plt.ioff()
    fig = plt.figure(figsize=(12, 9), dpi=200)
    axs = fig.subplots(
        8,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1]},
    )

    if warmup_span:
        for ax in axs:
            ax.axvspan(warmup_span[0], warmup_span[1], color="khaki", alpha=0.15, zorder=0)
        axs[0].plot([warmup_span[0]], [np.nan], color="khaki", alpha=0.3, linewidth=6, label="warmup")

    # Comfort band
    axs[0].plot(t_days_main, lower_c, "--", color="seagreen", linewidth=1, label="Comfort band")
    axs[0].plot(t_days_main, upper_c, "--", color="seagreen", linewidth=1)
    if has_warmup:
        axs[0].plot(warm_time, w_sp, "-", color="gray", linewidth=1, alpha=0.8, label="warmup setpoint")
        axs[0].plot(warm_time, warm_tz, "-", color="darkorange", alpha=0.8, label="warmup Tz")
        axs[0].plot(warm_time, w_lower_c, "--", color="seagreen", linewidth=1)
        axs[0].plot(warm_time, w_upper_c, "--", color="seagreen", linewidth=1)

    axs[0].step(t_days_rl, sp_rl, where="post", color="gray", linewidth=1, label="Setpoint")
    axs[0].plot(t_days_main, tz_c, "-", color="darkorange", linewidth=1, label="Tz")
    axs[0].set_ylabel("Tz / setpoint\n(°C)")

    # Bands: max price (red) + min occ (blue hatching), aligned with gymRC5
    if price.size:
        p_max = float(price.max())
        mask_max = np.isclose(price, p_max, rtol=1e-5, atol=1e-8)
        if mask_max.any():
            idx_max = np.where(mask_max)[0]
            start = idx_max[0]
            prev = idx_max[0]
            first_span_price = True
            for k in idx_max[1:]:
                if k == prev + 1:
                    prev = k
                else:
                    axs[0].axvspan(
                        t_days_main[start],
                        t_days_main[prev],
                        color="lightcoral",
                        alpha=0.18,
                        zorder=0,
                        label="Max price" if first_span_price else None,
                    )
                    first_span_price = False
                    start = prev = k
                prev = k
            axs[0].axvspan(
                t_days_main[start],
                t_days_main[prev],
                color="lightcoral",
                alpha=0.18,
                zorder=0,
                label="Max price" if first_span_price else None,
            )

    if prob.size:
        o_min = float(prob.min())
        mask_min = np.isclose(prob, o_min, rtol=1e-5, atol=1e-8)
        if mask_min.any():
            idx_min = np.where(mask_min)[0]
            start = idx_min[0]
            prev = idx_min[0]
            first_span_occ = True
            for k in idx_min[1:]:
                if k == prev + 1:
                    prev = k
                else:
                    axs[0].axvspan(
                        t_days_main[start],
                        t_days_main[prev],
                        facecolor="cornflowerblue",
                        edgecolor="cornflowerblue",
                        alpha=0.25,
                        hatch="//",
                        zorder=0,
                        label="Occ. min" if first_span_occ else None,
                    )
                    first_span_occ = False
                    start = prev = k
                prev = k
            axs[0].axvspan(
                t_days_main[start],
                t_days_main[prev],
                facecolor="cornflowerblue",
                edgecolor="cornflowerblue",
                alpha=0.25,
                hatch="//",
                zorder=0,
                label="Occ. min" if first_span_occ else None,
            )

    axs[0].legend(fontsize=7)

    axs[1].plot(t_days_main, u_arr, "-", color="slateblue", linewidth=1)
    if has_warmup and warm_u.size:
        n = min(warm_u.size, warm_time.size)
        axs[1].plot(warm_time[:n], warm_u[:n], "-", color="slateblue", alpha=0.7)
    axs[1].set_ylabel("Commande\n(-)")

    axs[2].plot(t_days_main, php_arr, "-", color="black", linewidth=1, label="P_hp")
    if has_warmup and warm_php.size:
        axs[2].plot(warm_time, warm_php, "-", color="black", linewidth=1, alpha=0.7)
    axs[2].set_ylabel("P_hp (W)")
    axs[2].legend(loc="upper right", fontsize=7)

    axs[3].plot(t_days_rl, rewards, "b", linewidth=1, label=reward_label)
    if indiv is not None:
        axs[3].plot(t_days_rl, indiv[:, 0], "r", linewidth=1, label=comfort_label)
        axs[3].plot(t_days_rl, indiv[:, 1], "g", linewidth=1, label=energy_label)
        axs[3].plot(t_days_rl, indiv[:, 2], "m", linewidth=1, label=sat_label)
    axs[3].set_ylabel("Rewards")
    axs[3].legend(loc="lower left", fontsize=7)

    axs[4].plot(t_days_main, ta, color="royalblue", linewidth=1, label="Ta")
    if has_warmup:
        axs[4].plot(warm_time, w_ta, "-", color="royalblue", linewidth=1, alpha=0.7)
    axq3 = axs[4].twinx()
    axq3.plot(t_days_main, qsol, color="gold", linewidth=1, label="Qsol")
    if has_warmup:
        axq3.plot(warm_time, w_qsol, "-", color="gold", linewidth=1, alpha=0.7)
    axs[4].set_ylabel("Ta (°C)")
    axq3.set_ylabel("Qsol (W)")
    axs[4].legend(loc="upper left", fontsize=7)
    axq3.legend(loc="upper right", fontsize=7)

    axs[5].plot(t_days_main, qocc, color="firebrick", linewidth=1, label="Qocc")
    axs[5].plot(t_days_main, qocr, color="darkred", linewidth=1, label="Qocr")
    if has_warmup:
        axs[5].plot(warm_time, w_qocc, "-", color="firebrick", linewidth=1, alpha=0.7)
        axs[5].plot(warm_time, w_qocr, "-", color="darkred", linewidth=1, alpha=0.7)
    axs[5].set_ylabel("Internal gains (W)")
    axs[5].legend(loc="upper right", fontsize=7)

    axs[6].step(t_days_main, occ, where="post", color="black", linewidth=1)
    axs[6].plot(t_days_main, prob, color="red", linewidth=0.8)
    if has_warmup:
        axs[6].step(warm_time, w_occ, where="post", color="black", linewidth=1, alpha=0.7)
        axs[6].plot(warm_time, w_prob, color="red", linewidth=0.8, alpha=0.7)
    axs[6].set_ylabel("Occup\n(-)")

    axs[7].plot(t_days_main, price, color="black", linewidth=1)
    if has_warmup:
        axs[7].plot(warm_time, w_price, "-", color="black", linewidth=1, alpha=0.7)
    axs[7].set_ylabel("Price\n(€/kWh)")
    axs[7].set_xlabel("Time (days)")

    fig.suptitle(f"MPC | consumption={energy_kwh:.1f} kWh")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    theta_scaled = scale_rc5_building(sim_opti_loaded.model.theta, K)
    model_scaled = eqx.tree_at(lambda m: m.theta, sim_opti_loaded.model, theta_scaled)
    sim_building = sim_opti_loaded.copy(model=model_scaled)

    if int(dataset_short.time.shape[0]) < 2:
        raise ValueError("dataset_short.time must contain at least two points.")
    dt_s = float(dataset_short.time[1] - dataset_short.time[0])

    nZOH = max(1, int(round(MPC_STEP_S / dt_s)))
    window_size = max(2, int(round((MPC_HORIZON_H * 3600.0) / dt_s)))

    t_warm, d_warm = _window(dataset_short, START_TIME_S - WARMUP_DAYS * 86400, START_TIME_S)
    t_ep, d_ep = _window(dataset_short, START_TIME_S, START_TIME_S + EPISODE_DAYS * 86400)

    theta = sim_building.model.theta
    x0_warmup = RC5_steady_state_sys(
        float(d_warm["weaSta_reaWeaTDryBul_y"][0]),
        float(d_warm["weaSta_reaWeaHGloHor_y"][0]),
        float(d_warm["InternalGainsCon[1]"][0]),
        float(d_warm["InternalGainsRad[1]"][0]),
        float(d_warm["reaQHeaPumCon_y"][0]),
        theta,
    )

    sim_warmup = sim_building.copy(
        x0=x0_warmup,
        time_grid=t_warm,
        d=d_warm,
        integrator=INTEGRATOR,
    )
    sp_warmup = jnp.full((t_warm.shape[0],), BASE_SETPOINT_K, dtype=jnp.float64)
    pid = Controller_PID(k_p=0.6, k_i=0.6 / 800.0, k_d=0.0, n=1, verbose=False, SetPoints=sp_warmup)
    t_w, y_w, x_w, u_w, *_ = sim_warmup.run_numpy(controller=pid)
    x0_mpc = x_w[-1]

    setpoints = 0.5 * (
        jnp.asarray(d_ep["LowerSetp[1]"], dtype=jnp.float64)
        + jnp.asarray(d_ep["UpperSetp[1]"], dtype=jnp.float64)
    )

    sim_mpc = sim_building.copy(
        x0=x0_mpc,
        time_grid=t_ep,
        d=d_ep,
        integrator=INTEGRATOR,
    )
    mpc = Controller_MPC(
        sim=sim_mpc,
        window_size=window_size,
        n=nZOH,
        W=W,
        SetPoints=setpoints,
        cost_core=cost_core,
        verbose=True,
    )
    t, y, _x, u, _ctrl_states, mpc_logs = sim_mpc.copy(controller=mpc).run_numpy()

    _plot(
        warmup=(t_w, y_w, u_w, d_warm),
        episode=(t, y, u, d_ep, setpoints),
        forecasts=mpc_logs,
        nZOH=nZOH,
        out_path=OUT_PATH,
    )


if __name__ == "__main__":
    main()
