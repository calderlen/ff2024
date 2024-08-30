import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import os as os

class modelEvaluation:

    def __init__(self, trained_models, datasets):
        self.trained_models = trained_models
        self.datasets = datasets

    # Permutation importance
    
    @staticmethod
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
    
    @staticmethod
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
    @staticmethod
    def calculate_metrics(model, X_train, y_train, X_test, y_test):

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


    def feature_importance(self):
        for key, models in self.trained_models.items():
            for i, model in enumerate(models):
                dataset = self.datasets[key][i]
                print(f"Evaluating RNN model for {key} - split {i + 1}...")
                self.feature_importance(model, dataset['X_train'], dataset['y_train'], dataset['X_test'], dataset['y_test'])
                importances = self.compute_permutation_importance(model, dataset['X_test'], dataset['y_test'])
                self.plot_feature_importance(importances, feature_names=dataset['feature_names'], title=f"{model.name}_feature_importance")