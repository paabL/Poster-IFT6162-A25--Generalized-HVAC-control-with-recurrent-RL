from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from gymRC5 import MyMinimalEnv, NormalizeAction, ResidualActionWrapper
from gymRC5_lstm import MyMinimalEnvLSTM
from Utils.rc5_multi_theta import KModelWrapper, build_k_models


MODEL_PATHS = [
    Path("Pre_ppo_rc5_1FA_LSTM_HU.zip"),
]
VECNORM_PATH = Path("vecnormalize_stats_1FA_LSTM_HU.pkl")

PLOT_PER_KS = True
KS_INDICES: list[int] | None = None  # ex: [0, 3] ; None = tous

EPISODE_START_TIME_S = 28 * 24 * 3600  # 1er février si t=0 = 1er janvier

N_EPISODES = 1
DETERMINISTIC = True
DEVICE = "cpu"
MAX_STEPS: int | None = None  # cap steps RL (debug)

# Baseline: agent constant (consigne fixe)
constant = True
CONSTANT_VALUE_C = 22.0  # consigne (°C) (base_setpoint du script)
KEEP_PLOTS_OPEN = False  # sinon les fenêtres peuvent se fermer à la fin du script
SAVE_PLOTS = False      # utile si backend headless (Agg)
AUTO_OPEN_SAVED_PLOTS = False  # robuste: ouvre le dossier `plots/`
PLOTS_DIR = Path("plots")

# IMPORTANT: doit matcher le modèle/VecNormalize (sinon obs_space mismatch)
# Entraînement (`1FA_LSTM.py`) : MyMinimalEnvLSTM avec past_steps=0 (Dict obs: {now, forecast})
PAST_STEPS = 0
FUTURE_STEPS = 12
WARMUP_STEPS = 4 * 24
MAX_EPISODE_LENGTH = 24 * 7
EXCLUDING_PERIODS = None


KS_PRESETS = [
    #{"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    #{"k_size": 0.8, "k_U": 0.8, "k_inf": 0.8, "k_win": 1.0, "k_mass": 0.9},
    #{"k_size": 1.2, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.1},
    #{"k_size": 1.0, "k_U": 0.7, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    #{"k_size": 1.0, "k_U": 1.3, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    #{"k_size": 1.0, "k_U": 1.0, "k_inf": 0.7, "k_win": 1.0, "k_mass": 1.0},
    #{"k_size": 1.0, "k_U": 1.0, "k_inf": 1.3, "k_win": 1.0, "k_mass": 1.0},
    #{"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.2, "k_mass": 1.0},
    #{"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 0.8, "k_mass": 1.0},
    #{"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.2},
    {"k_size": 1.1, "k_U": 0.9, "k_inf": 1.1, "k_win": 0.9, "k_mass": 1.1}, #Pas dans l'entrainement
]


def _unwrap_to(env, target_type):
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, target_type):
            return cur
        seen.add(id(cur))
        cur = getattr(cur, "env", None)
    return None


def _make_venv(
    *,
    n_envs: int,
    fixed_model_idx: int | None,
    seed: int,
    render_episodes: bool,
    past_steps: int,
    future_steps: int,
    warmup_steps: int,
    max_episode_length: int,
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    base_sp = 273.15 + 21.0

    thetas = build_k_models(KS_PRESETS)

    def _make_one(rank: int = 0):
        env = MyMinimalEnvLSTM(
            step_period=3600,
            past_steps=past_steps,
            future_steps=future_steps,
            warmup_steps=warmup_steps,
            base_setpoint=base_sp,
            render_episodes=render_episodes,
            max_episode_length=max_episode_length,
            excluding_periods=EXCLUDING_PERIODS,
        )
        env = KModelWrapper(
            env,
            thetas=thetas,
            ks=KS_PRESETS,
            seed=seed + rank,
            fixed_model_idx=fixed_model_idx,
        )
        env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
        env = NormalizeAction(env)
        env = Monitor(env)
        return env

    if n_envs == 1:
        return DummyVecEnv([lambda: _make_one(0)])
    return DummyVecEnv([lambda r=i: _make_one(r) for i in range(n_envs)])


def _load_vecnormalize(venv, vecnorm_path: Path | None):
    if vecnorm_path is None:
        return venv
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats introuvables: {vecnorm_path}")
    from stable_baselines3.common.vec_env import VecNormalize

    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def _load_model(model_path: Path, *, venv, device: str):
    from sb3_contrib import RecurrentPPO

    return RecurrentPPO.load(str(model_path), env=venv, device=device)


def _rollout_one_episode(*, model, venv, deterministic: bool, max_steps: int | None):
    obs = venv.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _info = obs
    ep_return = 0.0
    ep_len = 0

    state = None
    episode_starts = np.ones((venv.num_envs,), dtype=bool)

    while True:
        if max_steps is not None and ep_len >= max_steps:
            return {"return": ep_return, "len": ep_len, "monitor": {}}

        try:
            action, state = model.predict(
                obs,
                state=state,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        except TypeError:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, dones, infos = venv.step(action)
        ep_return += float(np.asarray(reward)[0])
        ep_len += 1
        episode_starts = np.asarray(dones, dtype=bool)
        if bool(np.asarray(dones)[0]):
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            ep_info = dict(info0.get("episode", {})) if isinstance(info0, dict) else {}
            return {"return": ep_return, "len": ep_len, "monitor": ep_info}

def _rollout_one_episode_single_env(*, model, env, vecnorm, deterministic: bool, max_steps: int | None, start_time_s: float):
    obs, _info = env.reset(seed=0, options={"start_time_s": float(start_time_s)})
    ep_return = 0.0
    ep_len = 0
    state = None
    episode_start = np.array([True], dtype=bool)

    while True:
        if max_steps is not None and ep_len >= max_steps:
            return {"return": ep_return, "len": ep_len, "monitor": {}}

        obs_norm = vecnorm.normalize_obs(obs)
        action, state = model.predict(obs_norm, state=state, episode_start=episode_start, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1
        done = bool(terminated or truncated)
        episode_start[...] = done
        if done:
            ep_info = dict(info.get("episode", {})) if isinstance(info, dict) else {}
            return {"return": ep_return, "len": ep_len, "monitor": ep_info}

def _rollout_one_episode_single_env_constant(*, env, max_steps: int | None, start_time_s: float):
    obs, _info = env.reset(seed=0, options={"start_time_s": float(start_time_s)})
    ep_return = 0.0
    ep_len = 0
    action = np.zeros(env.action_space.shape, dtype=np.float32)  # 0 => base_setpoint (=22°C)

    while True:
        if max_steps is not None and ep_len >= max_steps:
            return {"return": ep_return, "len": ep_len, "monitor": {}}

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1
        done = bool(terminated or truncated)
        if done:
            ep_info = dict(info.get("episode", {})) if isinstance(info, dict) else {}
            return {"return": ep_return, "len": ep_len, "monitor": ep_info}


def _plot_last_episode(venv) -> None:
    cur = venv
    while hasattr(cur, "venv"):
        cur = cur.venv
    base = _unwrap_to(cur.envs[0], MyMinimalEnv)
    if base is None:
        raise RuntimeError("Impossible de retrouver MyMinimalEnv (wrappers inattendus).")
    base.rollout_dir = PLOTS_DIR
    base._plot_episode()


def _open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)


if __name__ == "__main__":
    model_paths = [p for p in MODEL_PATHS if p.exists()]
    if not model_paths:
        raise SystemExit("Aucun modèle trouvé (edite `MODEL_PATHS`).")

    ks_indices = KS_INDICES if KS_INDICES is not None else list(range(len(KS_PRESETS)))
    if not PLOT_PER_KS:
        ks_indices = [None]

    for model_path in model_paths:
        print(f"\n=== {model_path} ===")
        for ks_idx in ks_indices:
            venv_norm = _make_venv(
                n_envs=1,
                fixed_model_idx=ks_idx,
                seed=0,
                render_episodes=False,
                past_steps=PAST_STEPS,
                future_steps=FUTURE_STEPS,
                warmup_steps=WARMUP_STEPS,
                max_episode_length=MAX_EPISODE_LENGTH,
            )
            venv_norm = _load_vecnormalize(venv_norm, VECNORM_PATH)
            model = _load_model(model_path, venv=venv_norm, device=DEVICE)

            env = venv_norm.venv.envs[0]  # même wrappers, mais sans auto-reset VecEnv

            if ks_idx is not None:
                print(f"KS[{ks_idx}]={KS_PRESETS[int(ks_idx)]}")

            for ep in range(N_EPISODES):
                out = _rollout_one_episode_single_env(
                    model=model,
                    env=env,
                    vecnorm=venv_norm,
                    deterministic=DETERMINISTIC,
                    max_steps=MAX_STEPS,
                    start_time_s=EPISODE_START_TIME_S,
                )
                mon = out.get("monitor", {}) or {}
                r = mon.get("r", out["return"])
                l = mon.get("l", out["len"])
                print(f"[rl] ep={ep+1}/{N_EPISODES} return={r:.3f} len={int(l)}")
                base = _unwrap_to(env, MyMinimalEnv)
                if base is None:
                    raise RuntimeError("Impossible de retrouver MyMinimalEnv (wrappers inattendus).")
                base.rollout_dir = PLOTS_DIR
                base._plot_episode()
                if SAVE_PLOTS:
                    import matplotlib.pyplot as plt

                    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                    ks_tag = "all" if ks_idx is None else f"ks{int(ks_idx)}"
                    out_path = PLOTS_DIR / f"{model_path.stem}_{ks_tag}_ep{ep+1}.png"
                    fig = plt.gcf()
                    fig.canvas.draw()
                    fig.savefig(out_path, dpi=150)

                if constant:
                    out_c = _rollout_one_episode_single_env_constant(
                        env=env,
                        max_steps=MAX_STEPS,
                        start_time_s=EPISODE_START_TIME_S,
                    )
                    mon_c = out_c.get("monitor", {}) or {}
                    r_c = mon_c.get("r", out_c["return"])
                    l_c = mon_c.get("l", out_c["len"])
                    print(f"[const={CONSTANT_VALUE_C:g}C] ep={ep+1}/{N_EPISODES} return={r_c:.3f} len={int(l_c)}")
                    base = _unwrap_to(env, MyMinimalEnv)
                    if base is None:
                        raise RuntimeError("Impossible de retrouver MyMinimalEnv (wrappers inattendus).")
                    base.rollout_dir = PLOTS_DIR
                    base._plot_episode()

            env.close()
            venv_norm.close()

    if KEEP_PLOTS_OPEN:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show(block=True)

    if SAVE_PLOTS and AUTO_OPEN_SAVED_PLOTS:
        _open_path(PLOTS_DIR)
