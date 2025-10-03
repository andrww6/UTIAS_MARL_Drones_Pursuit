
STATIC_ENCRYPT_LEN = 100

TRAIN_CONFIG_PARAMS = ["num_env", "num_iter", "train_timesteps", "val_iter", "val_num", "if_render", "val_save_step",
                       "val_save_path", "population_size", "iter_per_selection"]

ENV_CONFIG_PARAMS = ["mode", "num_p", "num_e", "r_velocity", "delta_t", "r_perception", "algorithm"]

PPO_AGENT_CONFIG_PARAMS = ["type", "policy", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                           "gae_lambda",
                           "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "verbose", "device"]

DQN_AGENT_CONFIG_PARAMS = ["type", "policy", "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau",
                           "gamma", "train_freq", "gradient_steps", "target_update_interval", "exploration_fraction",
                           "exploration_initial_eps", "exploration_final_eps", "max_grad_norm", "stats_window_size",
                           "verbose", "device"]