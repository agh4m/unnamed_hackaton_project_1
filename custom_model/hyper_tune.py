import random
import itertools
import concurrent.futures
import rf_classification

# Hyperparameter ranges for grid search (randomized order)
param_ranges = {
    'lr': [0.0001, 0.001, 0.01],
    'wd': [1e-5, 1e-4, 1e-3],
    'dropout': [0.3, 0.5, 0.7],
    'batch_size': [16, 32],
    'step_size': [5, 10, 15],
    'gamma': [0.1, 0.5],
    'max_samples': [50, 100],
    'noise_level': [0.05, 0.1, 0.2],
    'shift_max': [25, 50],
    'num_epochs': [10],
    'k': [3],
    'patience': [5],
}

# Generate all combinations and shuffle
param_combos = list(itertools.product(*param_ranges.values()))
random.shuffle(param_combos)
max_workers = 4

num_trials = min(15, len(param_combos))  # Test up to 15 unique combos


def run_trial(combo):
    params = dict(zip(param_ranges.keys(), combo))
    print(f"Running trial with params: {params}")
    try:
        acc = rf_classification.train_model(params)
        print(f"Completed trial, accuracy: {acc:.4f}")
        return acc, params
    except Exception as e:
        print(f"Error in trial: {e}")
        return 0, params


# Output info
print(f"Total unique combinations: {len(param_combos)}")
k = param_ranges['k'][0]
num_epochs = param_ranges['num_epochs'][0]
time_per_trial_min = k * num_epochs * 2.5
total_time_min = num_trials * time_per_trial_min
print(f"Estimated time for {num_trials} trials (sequential): {
      total_time_min} minutes ({total_time_min / 60:.2f} hours)")
print(f"With {max_workers} parallel workers, estimated time: {
      total_time_min / max_workers} minutes ({total_time_min / max_workers / 60:.2f} hours)")

best_acc = 0
best_params = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(
        run_trial, param_combos[i]): i for i in range(num_trials)}
    for future in concurrent.futures.as_completed(futures):
        trial_idx = futures[future]
        try:
            acc, params = future.result()
            if acc > best_acc:
                best_acc = acc
                best_params = params.copy()
        except Exception as e:
            print(f"Trial {trial_idx + 1} generated an exception: {e}")

print(f"Best accuracy: {best_acc:.4f}")
print(f"Best params: {best_params}")
