

steps_per_epoch = 6000
epochs = 12000
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 1e-4
value_function_learning_rate = 1e-4
train_policy_iterations = 40
train_value_iterations = 40
lam = 0.97
target_kl = 0.01
hidden_sizes =  (128, 128, 128, 64, 64, 32, 32, 16)