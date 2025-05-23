# marketml/portfolio_opt/rl_optimizer.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

try:
    from marketml.configs import configs
    from marketml.portfolio_opt.rl_environment import PortfolioEnv
    from stable_baselines3 import PPO, A2C # DDPG bị loại bỏ để đơn giản
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback # StopTrainingOnRewardThreshold có thể không cần thiết ban đầu
    # from stable_baselines3.common.vec_env import SubprocVecEnv # Nếu muốn chạy song song nhiều env
except ImportError as e:
    print(f"LỖI NGHIÊM TRỌNG trong rl_optimizer.py: Không thể nhập các mô-đun cần thiết. {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

def train_rl_agent(
    prices_df_train: pd.DataFrame,
    financial_data_train: pd.DataFrame = None,
    classification_probs_train: pd.DataFrame = None,
    financial_features_list: list = None,
    prob_features_list: list = None,
    initial_capital=100000,
    transaction_cost_bps=10,
    lookback_window_size=30,
    rebalance_frequency_days=1,
    total_timesteps=100000,
    rl_algorithm="PPO",
    model_save_path: Path = None, # Đây là đường dẫn tới model.zip, scaler sẽ được lưu cùng thư mục
    log_dir: Path = None,
    ppo_n_steps=2048, ppo_batch_size=64, ppo_n_epochs=10, ppo_gamma=0.99,
    ppo_gae_lambda=0.95, ppo_clip_range=0.2, ppo_ent_coef=0.0, ppo_vf_coef=0.5,
    ppo_max_grad_norm=0.5, ppo_learning_rate=0.0003, ppo_policy_kwargs=None,
    reward_use_log_return=True, reward_turnover_penalty_factor=0.0
):
    logger.info(f"Bắt đầu huấn luyện tác nhân RL với {rl_algorithm}...")
    # ... (log siêu tham số) ...

    fin_feat_means = None
    fin_feat_stds = None
    if financial_data_train is not None and not financial_data_train.empty and financial_features_list:
        numeric_fin_features = []
        for feat in financial_features_list:
            # SỬA TÊN CỘT: 'Ticker' và 'Year' theo file financial_data.csv
            if feat in financial_data_train.columns and pd.api.types.is_numeric_dtype(financial_data_train[feat]):
                numeric_fin_features.append(feat)
            else:
                logger.warning(f"Đặc trưng tài chính '{feat}' không phải dạng số hoặc không tồn tại trong financial_data_train, sẽ bị bỏ qua khỏi tính toán mean/std.")
        
        if numeric_fin_features:
            temp_fin_data_for_stats = financial_data_train[numeric_fin_features].replace([np.inf, -np.inf], np.nan)
            fin_feat_means = temp_fin_data_for_stats.mean()
            fin_feat_stds = temp_fin_data_for_stats.std()
            if fin_feat_stds is not None:
                 fin_feat_stds[fin_feat_stds < 1e-6] = 1.0
            logger.info("Đã tính toán Mean và Std cho chuẩn hóa financial features (trên tập huấn luyện):")
            if fin_feat_means is not None: logger.info(f"Means:\n{fin_feat_means.to_string()}")
            if fin_feat_stds is not None: logger.info(f"Stds:\n{fin_feat_stds.to_string()}")

            # LƯU SCALERS
            if model_save_path is not None: # Chỉ lưu nếu model_save_path được cung cấp
                scaler_path = model_save_path.parent / "financial_scalers.joblib" # Lưu cùng thư mục với model chính
                try:
                    joblib.dump({'means': fin_feat_means, 'stds': fin_feat_stds}, scaler_path)
                    logger.info(f"Đã lưu financial feature scalers vào {scaler_path}")
                except Exception as e_save_scaler:
                    logger.error(f"Lỗi khi lưu financial scalers: {e_save_scaler}")
        else:
            logger.warning("Không có đặc trưng tài chính dạng số hợp lệ để tính mean/std.")

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
        'financial_features': financial_features_list,
        'prob_features': prob_features_list,
        'financial_feature_means': fin_feat_means,
        'financial_feature_stds': fin_feat_stds
    }
    
    # n_envs: Số môi trường chạy song song. Hiện tại là 1.
    # Có thể tăng lên nếu bạn có nhiều CPU cores và muốn tăng tốc huấn luyện (cần SubprocVecEnv)
    n_envs = 1
    vec_env = make_vec_env(PortfolioEnv, n_envs=n_envs, env_kwargs=env_kwargs, monitor_dir=str(log_dir) if log_dir else None)

    callbacks = []
    if model_save_path:
        # eval_freq: Đánh giá sau mỗi X bước. Số bước này nên là bội số của n_envs * ppo_n_steps / (số lần cập nhật mỗi rollout)
        # Hoặc đơn giản là một giá trị đủ lớn, ví dụ: total_timesteps // 20
        eval_freq_steps = max(5000, total_timesteps // 20) // n_envs # Phải chia cho n_envs
        eval_callback = EvalCallback(vec_env, # Nên dùng một môi trường đánh giá riêng biệt nếu có thể
                                     best_model_save_path=str(model_save_path.parent / "best_model"),
                                     log_path=str(model_save_path.parent / "eval_logs"),
                                     eval_freq=eval_freq_steps, # Số bước mỗi lần eval cho mỗi env
                                     n_eval_episodes=5, # Số episode để chạy đánh giá
                                     deterministic=True, render=False)
        callbacks.append(eval_callback)

    if rl_algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=ppo_n_epochs,
            gamma=ppo_gamma,
            gae_lambda=ppo_gae_lambda,
            clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef,
            vf_coef=ppo_vf_coef,
            max_grad_norm=ppo_max_grad_norm,
            learning_rate=ppo_learning_rate,
            policy_kwargs=ppo_policy_kwargs, # Truyền từ configs
            tensorboard_log=str(log_dir / "tb_logs") if log_dir else None,
            seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else None # Thêm seed
        )
    elif rl_algorithm.upper() == "A2C":
        # A2C thường dùng n_steps nhỏ hơn
        model = A2C(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=ppo_n_steps // 10 if ppo_n_steps > 50 else 5, # A2C n_steps thường nhỏ
            gamma=ppo_gamma,
            gae_lambda=ppo_gae_lambda, # A2C có thể không dùng gae_lambda trực tiếp tùy vào implement
            ent_coef=ppo_ent_coef,
            vf_coef=ppo_vf_coef,
            max_grad_norm=ppo_max_grad_norm,
            learning_rate=ppo_learning_rate,
            policy_kwargs=ppo_policy_kwargs,
            tensorboard_log=str(log_dir / "tb_logs") if log_dir else None,
            seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else None # Thêm seed
        )
    else:
        logger.error(f"Thuật toán RL không được hỗ trợ: {rl_algorithm}")
        vec_env.close()
        return None

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None, progress_bar=True) # Thêm progress_bar
        if model_save_path:
            final_model_path = model_save_path.parent / f"{rl_algorithm.lower()}_final_model_{total_timesteps}steps.zip"
            model.save(final_model_path)
            logger.info(f"Huấn luyện mô hình RL hoàn tất. Mô hình cuối cùng được lưu vào {final_model_path}")
            if any(isinstance(cb, EvalCallback) for cb in callbacks):
                 logger.info(f"Mô hình tốt nhất trong quá trình đánh giá được lưu trong {model_save_path.parent / 'best_model'}")
        else:
            logger.info("Huấn luyện mô hình RL hoàn tất. Mô hình không được lưu vì không có đường dẫn được cung cấp.")
        return model
    except Exception as e:
        logger.error(f"Lỗi trong quá trình huấn luyện mô hình RL: {e}", exc_info=True)
        return None
    finally:
        vec_env.close() # Luôn đóng môi trường

# --- Ví dụ về Optuna ---
# def objective(trial: optuna.Trial) -> float:
#     # Định nghĩa không gian siêu tham số để Optuna tìm kiếm
#     learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
#     n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
#     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
#     ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
#     # ... các siêu tham số khác ...
#
#     # Tạo prices_df_train, log_dir_trial, model_save_path_trial
#     # prices_df_for_training_opt = ...
#     # temp_log_dir = configs.RL_LOG_DIR / f"optuna_trial_{trial.number}"
#     # temp_model_save_path = temp_log_dir / "optuna_model.zip" # Không cần lưu mỗi trial
#
#     model = train_rl_agent(
#         prices_df_train=prices_df_for_training_opt, # Cần truyền dữ liệu vào đây
#         total_timesteps=50000, # Số bước huấn luyện cho mỗi trial (nên nhỏ hơn huấn luyện chính)
#         rl_algorithm="PPO",
#         model_save_path=None, # Không lưu model trong quá trình optimize siêu tham số
#         log_dir=temp_log_dir,
#         ppo_learning_rate=learning_rate,
#         ppo_n_steps=n_steps,
#         ppo_batch_size=batch_size,
#         ppo_ent_coef=ent_coef,
#         # ... các siêu tham số khác ...
#     )
#
#     if model is None:
#         return -float('inf') # Trả về giá trị tệ nếu huấn luyện thất bại
#
#     # Đánh giá mô hình: Tạo một môi trường đánh giá và chạy vài episode
#     eval_env_kwargs = { ... } # Tương tự như train_env_kwargs
#     eval_env = make_vec_env(PortfolioEnv, n_envs=1, env_kwargs=eval_env_kwargs)
#     mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
#     eval_env.close()
#     logger.info(f"Trial {trial.number}: Mean Reward = {mean_reward:.4f} +/- {std_reward:.4f}")
#     return mean_reward # Optuna sẽ cố gắng tối đa hóa giá trị này
#
# # Để chạy Optuna:
# # import optuna
# # from stable_baselines3.common.evaluation import evaluate_policy
# # study = optuna.create_study(direction="maximize")
# # study.optimize(objective, n_trials=50) # Số lần thử
# # print("Best trial:")
# # trial = study.best_trial
# # print(f"  Value: {trial.value}")
# # print("  Params: ")
# # for key, value in trial.params.items():
# #     print(f"    {key}: {value}")

def predict_rl_weights(model, current_env_state: np.ndarray, deterministic=True):
    """
    Lấy hành động (tỷ trọng) từ mô hình RL đã huấn luyện.
    current_env_state: Quan sát trạng thái hiện tại từ môi trường.
    """
    if model is None:
        logger.error("Mô hình RL chưa được tải. Không thể dự đoán tỷ trọng.")
        num_elements_in_action = model.action_space.shape[0] if hasattr(model, 'action_space') else 2
        return np.ones(num_elements_in_action) / num_elements_in_action


    action, _states = model.predict(current_env_state, deterministic=deterministic)
    
    predicted_weights = np.exp(action) / np.sum(np.exp(action))
    if isinstance(predicted_weights, list):
        predicted_weights = np.array(predicted_weights).flatten()
    elif predicted_weights.ndim > 1:
        predicted_weights = predicted_weights.flatten()
    logger.debug(f"RL Dự đoán hành động thô: {action}, Tỷ trọng Softmax: {predicted_weights}")
    return predicted_weights