# train_rc5_lstm.py
import random
from pathlib import Path
import numpy as np
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from gymRC5 import ResidualActionWrapper, NormalizeAction
from gymRC5_lstm import MyMinimalEnvLSTM
from Utils.rc5_multi_theta import KModelWrapper, build_k_models
from features_extractor_rc5 import StepMLPTimeWeightedFuseExtractor


CFG = dict(
    reload=False,
    n_envs=4,
    seed=0,
    fixed_model_idx=None,
    total_timesteps=5_000_000,
    model_path="Pre_ppo_rc5_1FA_LSTM_VHE.zip",
    vecnorm_path="vecnormalize_stats_1FA_LSTM_VHE.pkl",
)

ENV_CFG = dict(
    step_period=3600,
    past_steps=0,
    future_steps=12,
    warmup_steps=4 * 24,
    base_setpoint=273.15 + 21.0,
    # reward = -(w_energy*energy(€) + w_comfort*comfort(K·h) + w_sat*sat(unit·h))
    # Increasing `w_comfort` => more comfort (fewer violations), often more energy.
    # Increasing `w_energy`  => less energy, often more discomfort.
    w_energy=4.0,
    w_comfort=1.0, 
    # Smooths the comfort penalty near 0 (Huber, in Kelvin).
    # If comfort_huber_k > 0, small violations are penalized less (avoids the agent being "afraid" to get close).
    comfort_huber_k=0.5,
    w_sat=0.2,
    w_u=1.0/2*0, # Not used
    w_tz=1.0/(273.15*5)*0, # Not used
    render_episodes=True,
    max_episode_length=24 * 7,
    excluding_periods=[(28 * 24 * 3600, 36 * 24 * 3600)],
)

VECNORM_CFG = dict(
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
)

EXTRACTOR_CFG = dict(
    d_step=32,
    hidden_step=64,
    d_out=64,
    hidden_fuse=128,
    gamma=0.9,
)

PPO_LR_START = 1e-4
PPO_LR_END = 5e-5
SAVE_EVERY_STEPS = 1_000_000


def lr_schedule(progress_remaining: float) -> float:
    return PPO_LR_END + (PPO_LR_START - PPO_LR_END) * progress_remaining


PPO_CFG = dict(
    learning_rate=lr_schedule,
    n_steps=256,
    batch_size=256,
    n_epochs=3, # default is 10
    clip_range=0.2,
    target_kl=0.03,
    max_grad_norm=0.5,
    clip_range_vf=0.1,
    gae_lambda=0.90,
    device="cpu",
    verbose=1,
    tensorboard_log="tensorboard_logs",
)

KS = [
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 0.9, "k_U": 0.9, "k_inf": 0.9, "k_win": 1.0, "k_mass": 0.95},
    {"k_size": 1.1, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.05},
    {"k_size": 1.0, "k_U": 0.85, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.15, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 0.85, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.15, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.1, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 0.9, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.1},
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(rank: int, thetas):
    def _init():
        env = MyMinimalEnvLSTM(**ENV_CFG)
        env = KModelWrapper(
            env,
            thetas=thetas,
            ks=KS,
            seed=CFG["seed"] + rank,
            fixed_model_idx=CFG["fixed_model_idx"],
        )
        env = ResidualActionWrapper(env, base_action=ENV_CFG["base_setpoint"], max_dev=5.0)
        env = NormalizeAction(env)
        env = Monitor(env)
        return env

    return _init


def build_model(env, policy_kwargs):
    return RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=PPO_CFG["verbose"],
        learning_rate=PPO_CFG["learning_rate"],
        n_steps=PPO_CFG["n_steps"],
        batch_size=PPO_CFG["batch_size"],
        n_epochs=PPO_CFG["n_epochs"],
        clip_range=PPO_CFG["clip_range"],
        target_kl=PPO_CFG["target_kl"],
        max_grad_norm=PPO_CFG["max_grad_norm"],
        clip_range_vf=PPO_CFG["clip_range_vf"],
        gae_lambda=PPO_CFG["gae_lambda"],
        device=PPO_CFG["device"],
        tensorboard_log=PPO_CFG["tensorboard_log"],
    )

def make_save_cb(*, every_steps: int, model_path: str, venv, vecnorm_path: str):
    next_save = [every_steps]

    def _cb(_locals, _globals):
        if _locals["self"].num_timesteps >= next_save[0]:
            _locals["self"].save(model_path)
            venv.save(vecnorm_path)
            next_save[0] += every_steps
        return True

    return _cb


if __name__ == "__main__":
    set_global_seed(CFG["seed"])
    thetas = build_k_models(KS)

    venv = DummyVecEnv([make_env(i, thetas) for i in range(CFG["n_envs"])])

    if CFG["reload"]:
        venv = VecNormalize.load(CFG["vecnorm_path"], venv)
        venv.training = True
        venv.norm_obs = VECNORM_CFG["norm_obs"]
        venv.norm_reward = VECNORM_CFG["norm_reward"]

        old = RecurrentPPO.load(CFG["model_path"], env=venv, device=PPO_CFG["device"])
        model = build_model(venv, old.policy_kwargs)
        model.set_parameters(old.get_parameters(), exact_match=True)

        tb_log_name = "PPO_RC5_LSTM_continue"
        run_id = get_latest_run_id(PPO_CFG["tensorboard_log"], tb_log_name) + 1
        rollout_dir = Path(PPO_CFG["tensorboard_log"]) / f"{tb_log_name}_{run_id}" / "rollout"
        venv.env_method("set_rollout_dir", str(rollout_dir))

        model.learn(
            total_timesteps=CFG["total_timesteps"],
            tb_log_name=tb_log_name,
            reset_num_timesteps=False,
            callback=make_save_cb(
                every_steps=SAVE_EVERY_STEPS,
                model_path=CFG["model_path"],
                venv=venv,
                vecnorm_path=CFG["vecnorm_path"],
            ),
        )
    else:
        venv = VecNormalize(venv, **VECNORM_CFG)

        policy_kwargs = dict(
            features_extractor_class=StepMLPTimeWeightedFuseExtractor,
            features_extractor_kwargs=EXTRACTOR_CFG,
        )

        model = build_model(venv, policy_kwargs)

        with torch.no_grad():
            actor_net = model.policy.action_net
            actor_net.weight.fill_(0.0)
            actor_net.bias.fill_(0.0)
            if hasattr(model.policy, "log_std"):
                model.policy.log_std.data.fill_(-1.0)

        tb_log_name = "PPO_RC5_LSTM"
        run_id = get_latest_run_id(PPO_CFG["tensorboard_log"], tb_log_name) + 1
        rollout_dir = Path(PPO_CFG["tensorboard_log"]) / f"{tb_log_name}_{run_id}" / "rollout"
        venv.env_method("set_rollout_dir", str(rollout_dir))

        model.learn(
            total_timesteps=CFG["total_timesteps"],
            tb_log_name=tb_log_name,
            callback=make_save_cb(
                every_steps=SAVE_EVERY_STEPS,
                model_path=CFG["model_path"],
                venv=venv,
                vecnorm_path=CFG["vecnorm_path"],
            ),
        )
        model.save(CFG["model_path"])
        venv.save(CFG["vecnorm_path"])

    venv.close()
