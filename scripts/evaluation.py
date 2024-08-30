import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import os as os
import pandas as pd

class evaluation:
    def __init__(self, trained_models, datasets):
        self.trained_models = trained_models
        self.datasets = datasets
    # Permutation importance
    def compute_permutation_importance(model, X_test, y_test, n_repeats=5):

        baseline_mse = mean_squared_error(y_test, model.predict(X_test))
        importances = np.zeros(X_test.shape[2])

        for i in range(X_test.shape[2]):  # Loop over features
            shuffled_mses = []
            for _ in range(n_repeats):
                X_test_permuted = X_test.copy()
                # Shuffle along the feature dimension (axis=2)
                X_test_permuted[:, :, i] = np.random.permutation(X_test_permuted[:, :, i])
                shuffled_mses.append(mean_squared_error(y_test, model.predict(X_test_permuted)))
            
            importances[i] = np.mean(shuffled_mses) - baseline_mse

        return importances


    # Compute permutation feature importance for each model and each split
    feature_importance_results = {key: [] for key in trained_models.keys()}

    for key, models in trained_models.items():
        for i, model in enumerate(models):
            dataset = datasets[key][i]
            X_test = dataset['X_test']
            y_test = dataset['y_test']

            print(f"X_test shape for {key} - split {i + 1}: {X_test.shape}")

            print(f"Computing feature importance for {key} - split {i + 1}...")
            importances = compute_permutation_importance(model, X_test, y_test)
            feature_importance_results[key].append(importances)

    def plot_feature_importance(importances, feature_names, title):

        pl.figure(figsize=(12, 6))
        sns.barplot(x=importances, y=feature_names)
        pl.title(title)
        pl.xlabel("Importance")
        pl.ylabel("Feature")
        pl.grid(True)
        pl.tight_layout()

        os.makedirs("results", exist_ok=True)
        pl.savefig(f"results/{title}.png")


    # Metrics
    def evaluate_model(model, X_train, y_train, X_test, y_test):

        mse_train = mean_squared_error(y_train, model.predict(X_train))
        mse_test = mean_squared_error(y_test, model.predict(X_test))

        mae_train = mean_absolute_error(y_train, model.predict(X_train))
        mae_test = mean_absolute_error(y_test, model.predict(X_test))

        r2_train = r2_score(y_train, model.predict(X_train))
        r2_test = r2_score(y_test, model.predict(X_test))

        print(f"Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}")
        print(f"Train MAE: {mae_train:.2f}, Test MAE: {mae_test:.2f}")

        pl.figure(figsize=(12, 6))
        pl.scatter(y_test, model.predict(X_test), alpha=0.5)
        pl.xlabel("True values")
        pl.ylabel("Predicted values")
        pl.title("True vs Predicted values")


        os.makedirs("results", exist_ok=True)
        pl.savefig(f"results/{model.name}_true_vs_predicted.png")