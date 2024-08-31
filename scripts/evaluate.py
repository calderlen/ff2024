# necessary imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as pl
import os
import pandas as pd

DATA_PATH = 'data/player_stats.csv'
RESULTS_PATH = 'results/'
MODEL_PATH = 'models/'
PREDICTIONS_PATH = os.path.join(RESULTS_PATH, 'predictions_2024.xlsx')

def calculate_metrics(model, X_train, y_train, X_test, y_test):

    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))

    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    mae_test = mean_absolute_error(y_test, model.predict(X_test))

    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))

    metrics = {
    'MSE': [mse_train, mse_test],
    'MAE': [mae_train, mae_test],
    'R2': [r2_train, r2_test]
    }

    pl.figure(figsize=(12, 6))
    pl.scatter(y_test, model.predict(X_test), alpha=0.5)
    pl.xlabel("True values")
    pl.ylabel("Predicted values")
    pl.title("True vs Predicted values")


    os.makedirs("results", exist_ok=True)
    pl.savefig(f"results/{model.name}_true_vs_predicted.png")


    return metrics

def plot_metrics(metrics):
    metrics_df = pd.DataFrame(metrics, index=['train', 'test'])
    metrics_df.plot(kind='bar', figsize=(12, 6))
    pl.ylabel("Value")
    pl.title("Model metrics")

    pl.savefig("results/model_metrics.png")

# write function that extracts the player_ids from each positional dataset. It then calculates the mean fantasy points per game predicted for the following season for each player separated by position. The function should return a dictionary with the mean fantasy points per game for each player separated by positon.
def evaluate_model(model_predictions, datasets):
    mean_fantasy_points = {key: {} for key in model_predictions.keys()}

    for key, predictions_list in model_predictions.items():
        all_player_ids = []
        all_predictions = []

        for i, predictions in enumerate(predictions_list):
            dataset = datasets[key][i]
            player_ids = dataset['player_ids']

            if predictions.ndim == 2:
                mean_points = predictions.mean(axis=1)
            else:
                raise ValueError(f"Prediction array for {key} at index {i} is not 2D.")
            
            all_player_ids.extend(player_ids)
            all_predictions.extend(mean_points)

        # Calculate mean fantasy points per player
        player_points = {}
        for player_id, points in zip(all_player_ids, all_predictions):
            if player_id not in player_points:
                player_points[player_id] = []
            player_points[player_id].append(points)

        for player_id, points in player_points.items():
            mean_fantasy_points[key][player_id] = sum(points) / len(points)
    
    return mean_fantasy_points

def rank_players(mean_fantasy_points):
    ranked_players = {key: [] for key in mean_fantasy_points.keys()}

    for key, player_points in mean_fantasy_points.items():
        sorted_players = sorted(player_points.items(), key=lambda x: x[1], reverse=True)
        ranked_players[key] = sorted_players


    # save the ranked players to a file
    with open(PREDICTIONS_PATH, 'w') as f:
        for key, players in ranked_players.items():
            f.write(f"{key}\n")
            for player_id, points in players:
                f.write(f"{player_id},{points}\n")
            f.write("\n")

    return ranked_players
