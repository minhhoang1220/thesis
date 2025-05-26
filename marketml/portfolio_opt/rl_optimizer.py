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
    model_save_path: Path = None, # Full path to the .zip file for the main model
    eval_callback_log_path: Path = None, # Separate path for EvalCallback logs/best_model
    tensorboard_log_path: Path = None, # Separate path for TensorBoard logs
    logger_instance: logging.Logger = None, # Optional logger instance
    # PPO specific params from configs, with defaults if not in configs
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
    # Reward shaping params
    reward_use_log_return: bool = True,
    reward_turnover_penalty_factor: float = 0.0
) -> tuple[object | None, FinancialFeatureScaler | None]: # Returns (trained_model, fitted_scaler)
    
    current_logger = logger_instance if logger_instance else logger
    current_logger.info(f"Starting RL agent training with algorithm: {rl_algorithm}")
    current_logger.info(f"Total timesteps: {total_timesteps}")
    # Log other important parameters if needed

    # 1. Initialize and Fit FinancialFeatureScaler
    fitted_scaler = FinancialFeatureScaler(feature_names=financial_features_list if financial_features_list else [])
    if financial_data_train is not None and not financial_data_train.empty and financial_features_list:
        current_logger.info("Fitting FinancialFeatureScaler on RL training data...")
        fitted_scaler.fit(financial_data_train, financial_features_list)
        if model_save_path: # Save scaler in the same directory as the primary model would be saved
            try:
                scaler_save_dir = model_save_path.parent
                fitted_scaler.save(scaler_save_dir) # FinancialFeatureScaler handles its own filename
                current_logger.info(f"Financial scaler saved in directory: {scaler_save_dir}")
            except Exception as e_save_scl:
                current_logger.error(f"Error saving fitted financial scaler: {e_save_scl}", exc_info=True)
    else:
        current_logger.info("No financial data or features for RL training, scaler will be empty (means=0, stds=1).")

    # 2. Prepare Environment Kwargs
    env_kwargs = {
        'prices_df': prices_df_train,
        'financial_data': financial_data_train,
        'classification_probs': classification_probs_train,
        'initial_capital': initial_capital,
        'transaction_cost_bps': transaction_cost_bps, # For env's internal reward logic
        'lookback_window_size': lookback_window_size,
        'rebalance_frequency_days': rebalance_frequency_days, # Env steps by this, but backtester rebalances at its own freq
        'max_steps_per_episode': None, # Let the data length and rebal_freq determine this
        'reward_use_log_return': reward_use_log_return,
        'reward_turnover_penalty_factor': reward_turnover_penalty_factor,
        'financial_features': financial_features_list if financial_features_list else [],
        'prob_features': prob_features_list if prob_features_list else [],
        'financial_feature_means': fitted_scaler.means, # Pass fitted means
        'financial_feature_stds': fitted_scaler.stds,   # Pass fitted stds
        'logger_instance': current_logger # Pass logger to env
    }
    
    # 3. Create Vectorized Environment
    # monitor_dir for SB3 is for CSVs of episode rewards, lengths etc.
    # tensorboard_log is for TensorBoard specific logs.
    monitor_log_path = str(eval_callback_log_path / "monitor_logs") if eval_callback_log_path else None
    if monitor_log_path: Path(monitor_log_path).mkdir(parents=True, exist_ok=True)

    n_envs = 1 # For simplicity, can be increased with SubprocVecEnv
    try:
        vec_env = make_vec_env(PortfolioEnv, n_envs=n_envs, env_kwargs=env_kwargs, monitor_dir=monitor_log_path)
    except Exception as e_make_env:
        current_logger.error(f"Failed to create RL vectorized environment: {e_make_env}", exc_info=True)
        return None, fitted_scaler # Return scaler even if env fails, as it might have been fitted


    # 4. Setup Callbacks
    callbacks = []
    if eval_callback_log_path and model_save_path: # model_save_path's parent is where best_model.zip goes
        best_model_save_dir = eval_callback_log_path / "best_model" # Specific dir for best_model from callback
        eval_log_dir_for_callback = eval_callback_log_path / "eval_logs_sb3" # Specific dir for callback's own logs

        eval_freq_steps = max(1024, total_timesteps // 50) // n_envs # Eval more frequently
        current_logger.info(f"EvalCallback configured: eval_freq={eval_freq_steps} steps (per env), "
                            f"best_model_save_path={best_model_save_dir}")
        try:
            eval_env_for_callback = make_vec_env(PortfolioEnv, n_envs=1, env_kwargs=env_kwargs, monitor_dir=None) # Separate env for eval
            eval_callback = EvalCallback(
                eval_env_for_callback,
                best_model_save_path=str(best_model_save_dir),
                log_path=str(eval_log_dir_for_callback),
                eval_freq=eval_freq_steps,
                n_eval_episodes=max(1, 5 // n_envs), # Ensure at least 1, total reasonable number
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        except Exception as e_eval_cb:
            current_logger.error(f"Failed to setup EvalCallback: {e_eval_cb}", exc_info=True)


    # 5. Initialize RL Model
    model = None
    algo_upper = rl_algorithm.upper()
    
    # Ensure policy_kwargs is a dict if None
    current_ppo_policy_kwargs = ppo_policy_kwargs if ppo_policy_kwargs is not None else {}


    if algo_upper == "PPO":
        model = PPO(
            "MlpPolicy", vec_env, verbose=0, # verbose=1 for more output during training
            n_steps=ppo_n_steps, batch_size=ppo_batch_size, n_epochs=ppo_n_epochs,
            gamma=ppo_gamma, gae_lambda=ppo_gae_lambda, clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef, vf_coef=ppo_vf_coef, max_grad_norm=ppo_max_grad_norm,
            learning_rate=ppo_learning_rate, policy_kwargs=current_ppo_policy_kwargs,
            tensorboard_log=str(tensorboard_log_path) if tensorboard_log_path else None,
            seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else None
        )
    elif algo_upper == "A2C":
        model = A2C( # A2C usually needs smaller n_steps
            "MlpPolicy", vec_env, verbose=0,
            n_steps=max(5, ppo_n_steps // 20), # A2C n_steps typically small, e.g., 5
            gamma=ppo_gamma, gae_lambda=ppo_gae_lambda, # gae_lambda might not be used by all A2C impls
            ent_coef=ppo_ent_coef, vf_coef=ppo_vf_coef, max_grad_norm=ppo_max_grad_norm,
            learning_rate=ppo_learning_rate, policy_kwargs=current_ppo_policy_kwargs,
            tensorboard_log=str(tensorboard_log_path) if tensorboard_log_path else None,
            seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else None
        )
    else:
        current_logger.error(f"Unsupported RL algorithm: {rl_algorithm}")
        vec_env.close()
        return None, fitted_scaler

    # 6. Train the Model
    try:
        current_logger.info(f"Training RL model ({algo_upper}) for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True
        )
        if model_save_path: # This is the path for the *final* model after full training
            model.save(model_save_path)
            current_logger.info(f"RL model training complete. Final model saved to: {model_save_path.resolve()}")
            if any(isinstance(cb, EvalCallback) for cb in callbacks):
                 current_logger.info(f"Best model during evaluation (if any) saved in: {eval_callback_log_path / 'best_model'}")
        else:
            current_logger.info("RL model training complete. Model not saved as model_save_path was not provided.")
        return model, fitted_scaler
    except Exception as e_learn:
        current_logger.error(f"Error during RL model training: {e_learn}", exc_info=True)
        return None, fitted_scaler # Return scaler even if model training fails
    finally:
        vec_env.close()
        if 'eval_env_for_callback' in locals() and eval_env_for_callback is not None: # Close eval_env if created
            eval_env_for_callback.close()


def predict_rl_weights(model, current_env_state: np.ndarray, deterministic: bool = True, logger_instance: logging.Logger = None) -> np.ndarray:
    current_logger = logger_instance if logger_instance else logger
    if model is None:
        current_logger.error("RL model is None. Cannot predict weights. Returning equal weights.")
        # Fallback to equal weights if model is not available
        # The number of elements in action space depends on how it was defined in PortfolioEnv
        # Assuming it's num_assets + 1 (for cash)
        if hasattr(model, 'action_space') and model.action_space is not None: # Check if model has action_space
            num_elements = model.action_space.shape[0]
        else: # Fallback if model or action_space is truly None
            num_elements = 2 # Default to a small number to avoid error, but this is a sign of bigger issue
            current_logger.warning(f"RL model or its action_space is None. Defaulting to {num_elements} elements for equal weight prediction.")
        return np.ones(num_elements, dtype=np.float32) / num_elements

    action, _ = model.predict(current_env_state, deterministic=deterministic)
    
    # Apply softmax to convert raw action (logits) to probabilities (weights)
    # Subtract max for numerical stability before exp
    exp_action = np.exp(action - np.max(action))
    predicted_weights = exp_action / np.sum(exp_action)
    
    # Ensure it's a 1D array
    if isinstance(predicted_weights, list):
        predicted_weights = np.array(predicted_weights).flatten()
    elif predicted_weights.ndim > 1:
        predicted_weights = predicted_weights.flatten()
        
    current_logger.log(logging.DEBUG - 5, # Very verbose
                       f"RL raw action: {action}, Softmax weights: "
                       f"{[f'{w:.3f}' for w in predicted_weights]}")
    return predicted_weights.astype(np.float32)