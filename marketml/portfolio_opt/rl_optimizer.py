# marketml/portfolio_opt/rl_optimizer.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

try:
    from marketml.configs import configs
    from marketml.portfolio_opt.rl_environment import PortfolioEnv
    from marketml.portfolio_opt.rl_scaler_handler import FinancialFeatureScaler
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError as e:
    print(f"CRITICAL ERROR in rl_optimizer.py: Could not import necessary modules. {e}")
    raise

logger = logging.getLogger(__name__)

def train_rl_agent(
    prices_df_train: pd.DataFrame,
    financial_data_train: pd.DataFrame = None,
    classification_probs_train: pd.DataFrame = None,
    financial_features_list: list = None,
    prob_features_list: list = None,
    initial_capital: float = 100000,
    transaction_cost_bps: int = 10,
    lookback_window_size: int = 30,
    rebalance_frequency_days: int = 1,
    total_timesteps: int = 100000,
    rl_algorithm: str = "PPO",
    model_save_path: Path = None, # Path to save the main model
    eval_callback_log_path: Path = None, # Path for EvalCallback logs/best_model
    tensorboard_log_path: Path = None, # Path for TensorBoard logs
    logger_instance: logging.Logger = None,
    ppo_n_steps: int = 2048,
    ppo_batch_size: int = 64,
    ppo_n_epochs: int = 10,
    ppo_gamma: float = 0.99,
    ppo_gae_lambda: float = 0.95,
    ppo_clip_range: float = 0.2,
    ppo_ent_coef: float = 0.0,
    ppo_vf_coef: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_learning_rate: float = 0.0003,
    ppo_policy_kwargs: dict = None,
    reward_use_log_return: bool = True,
    reward_turnover_penalty_factor: float = 0.0
) -> tuple[object | None, FinancialFeatureScaler | None]: # Returns (trained_model, fitted_scaler)
    
    # === Section: Logging and Scaler Initialization ===
    current_logger = logger_instance if logger_instance else logger
    current_logger.info(f"Starting RL agent training with algorithm: {rl_algorithm}")
    current_logger.info(f"Total timesteps: {total_timesteps}")

    fitted_scaler = FinancialFeatureScaler(feature_names=financial_features_list if financial_features_list else [])
    if financial_data_train is not None and not financial_data_train.empty and financial_features_list:
        current_logger.info("Fitting FinancialFeatureScaler on RL training data...")
        fitted_scaler.fit(financial_data_train, financial_features_list)
        if model_save_path:
            try:
                scaler_save_dir = model_save_path.parent
                fitted_scaler.save(scaler_save_dir)
                current_logger.info(f"Financial scaler saved in directory: {scaler_save_dir}")
            except Exception as e_save_scl:
                current_logger.error(f"Error saving fitted financial scaler: {e_save_scl}", exc_info=True)
    else:
        current_logger.info("No financial data or features for RL training, scaler will be empty (means=0, stds=1).")

    # === Section: Environment Preparation ===
    env_kwargs = {
        'prices_df': prices_df_train,
        'financial_data': financial_data_train,
        'classification_probs': classification_probs_train,
        'initial_capital': initial_capital,
        'transaction_cost_bps': transaction_cost_bps,
        'lookback_window_size': lookback_window_size,
        'rebalance_frequency_days': rebalance_frequency_days,
        'max_steps_per_episode': None,
        'reward_use_log_return': reward_use_log_return,
        'reward_turnover_penalty_factor': reward_turnover_penalty_factor,
        'financial_features': financial_features_list if financial_features_list else [],
        'prob_features': prob_features_list if prob_features_list else [],
        'financial_feature_means': fitted_scaler.means,
        'financial_feature_stds': fitted_scaler.stds,
        'logger_instance': current_logger
    }
    
    monitor_log_path = str(eval_callback_log_path / "monitor_logs") if eval_callback_log_path else None
    if monitor_log_path: Path(monitor_log_path).mkdir(parents=True, exist_ok=True)

    n_envs = 1
    try:
        vec_env = make_vec_env(PortfolioEnv, n_envs=n_envs, env_kwargs=env_kwargs, monitor_dir=monitor_log_path)
    except Exception as e_make_env:
        current_logger.error(f"Failed to create RL vectorized environment: {e_make_env}", exc_info=True)
        return None, fitted_scaler

    # === Section: Callbacks Setup ===
    callbacks = []
    if eval_callback_log_path and model_save_path:
        best_model_save_dir = eval_callback_log_path / "best_model"
        eval_log_dir_for_callback = eval_callback_log_path / "eval_logs_sb3"

        eval_freq_steps = max(1024, total_timesteps // 50) // n_envs
        current_logger.info(f"EvalCallback configured: eval_freq={eval_freq_steps} steps (per env), "
                            f"best_model_save_path={best_model_save_dir}")
        try:
            eval_env_for_callback = make_vec_env(PortfolioEnv, n_envs=1, env_kwargs=env_kwargs, monitor_dir=None)
            eval_callback = EvalCallback(
                eval_env_for_callback,
                best_model_save_path=str(best_model_save_dir),
                log_path=str(eval_log_dir_for_callback),
                eval_freq=eval_freq_steps,
                n_eval_episodes=max(1, 5 // n_envs),
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        except Exception as e_eval_cb:
            current_logger.error(f"Failed to setup EvalCallback: {e_eval_cb}", exc_info=True)

    # === Section: RL Model Initialization ===
    model = None
    algo_upper = rl_algorithm.upper()
    current_ppo_policy_kwargs = ppo_policy_kwargs if ppo_policy_kwargs is not None else {}

    if algo_upper == "PPO":
        model = PPO(
            "MlpPolicy", vec_env, verbose=0,
            n_steps=ppo_n_steps, batch_size=ppo_batch_size, n_epochs=ppo_n_epochs,
            gamma=ppo_gamma, gae_lambda=ppo_gae_lambda, clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef, vf_coef=ppo_vf_coef, max_grad_norm=ppo_max_grad_norm,
            learning_rate=ppo_learning_rate, policy_kwargs=current_ppo_policy_kwargs,
            tensorboard_log=str(tensorboard_log_path) if tensorboard_log_path else None,
            seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else None
        )
    elif algo_upper == "A2C":
        model = A2C(
            "MlpPolicy", vec_env, verbose=0,
            n_steps=max(5, ppo_n_steps // 20),
            gamma=ppo_gamma, gae_lambda=ppo_gae_lambda,
            ent_coef=ppo_ent_coef, vf_coef=ppo_vf_coef, max_grad_norm=ppo_max_grad_norm,
            learning_rate=ppo_learning_rate, policy_kwargs=current_ppo_policy_kwargs,
            tensorboard_log=str(tensorboard_log_path) if tensorboard_log_path else None,
            seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else None
        )
    else:
        current_logger.error(f"Unsupported RL algorithm: {rl_algorithm}")
        vec_env.close()
        return None, fitted_scaler

    # === Section: Model Training ===
    try:
        current_logger.info(f"Training RL model ({algo_upper}) for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True
        )
        if model_save_path:
            model.save(model_save_path)
            current_logger.info(f"RL model training complete. Final model saved to: {model_save_path.resolve()}")
            if any(isinstance(cb, EvalCallback) for cb in callbacks):
                 current_logger.info(f"Best model during evaluation (if any) saved in: {eval_callback_log_path / 'best_model'}")
        else:
            current_logger.info("RL model training complete. Model not saved as model_save_path was not provided.")
        return model, fitted_scaler
    except Exception as e_learn:
        current_logger.error(f"Error during RL model training: {e_learn}", exc_info=True)
        return None, fitted_scaler
    finally:
        vec_env.close()
        if 'eval_env_for_callback' in locals() and eval_env_for_callback is not None:
            eval_env_for_callback.close()

def predict_rl_weights(model, current_env_state: np.ndarray, deterministic: bool = True, logger_instance: logging.Logger = None) -> np.ndarray:
    # === Section: RL Weight Prediction ===
    current_logger = logger_instance if logger_instance else logger
    if model is None:
        current_logger.error("RL model is None. Cannot predict weights. Returning equal weights.")
        if hasattr(model, 'action_space') and model.action_space is not None:
            num_elements = model.action_space.shape[0]
        else:
            num_elements = 2
            current_logger.warning(f"RL model or its action_space is None. Defaulting to {num_elements} elements for equal weight prediction.")
        return np.ones(num_elements, dtype=np.float32) / num_elements

    action, _ = model.predict(current_env_state, deterministic=deterministic)
    exp_action = np.exp(action - np.max(action))
    predicted_weights = exp_action / np.sum(exp_action)
    if isinstance(predicted_weights, list):
        predicted_weights = np.array(predicted_weights).flatten()
    elif predicted_weights.ndim > 1:
        predicted_weights = predicted_weights.flatten()
    current_logger.log(logging.DEBUG - 5,
                       f"RL raw action: {action}, Softmax weights: "
                       f"{[f'{w:.3f}' for w in predicted_weights]}")
    return predicted_weights.astype(np.float32)
