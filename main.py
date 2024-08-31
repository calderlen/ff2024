import pandas as pd
import os
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import load_model

from scripts.dataProcessor import feature_engineering, clean_data, create_sequences, generate_cv_splits
from scripts.train import train_model
from scripts.test import test_model

DATA_PATH = 'data/player_stats.csv'
RESULTS_PATH = 'results/'
MODEL_PATH = 'models/'
PREDICTIONS_PATH = os.path.join(RESULTS_PATH, 'predictions_2024.xlsx')

df = pd.read_csv(DATA_PATH)
df = df.sort_values(by=['player_id', 'season', 'week'])

sequence_length = 3
tscv = TimeSeriesSplit(n_splits=5)

df = feature_engineering(df)

qb_df_features, qb_df_target, rb_df_features, rb_df_target, wr_df_features, wr_df_target, te_df_features, te_df_target = clean_data(df)

X_qb, y_qb = create_sequences(qb_df_features.values, qb_df_target.values, sequence_length)
X_rb, y_rb = create_sequences(rb_df_features.values, rb_df_target.values, sequence_length)
X_wr, y_wr = create_sequences(wr_df_features.values, wr_df_target.values, sequence_length)
X_te, y_te = create_sequences(te_df_features.values, te_df_target.values, sequence_length)

datasets = {'QB': [], 'RB': [], 'WR': [], 'TE': []}

for X, y, key in [(X_qb, y_qb, 'QB'), 
                  (X_rb, y_rb, 'RB'), 
                  (X_wr, y_wr, 'WR'), 
                  (X_te, y_te, 'TE')]:
    for i, (X_train, X_test, y_train, y_test) in enumerate(generate_cv_splits(X, y, tscv)):
        datasets[key].append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })

# Check if model folder and file exists, and if it doesn't create it and train a model. Otherwise, load the model and test it.
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

    trained_model = train_model(datasets)
else:
    trained_model = {key: [] for key in datasets.keys()}
    for key in datasets.keys():
        for i in range(5):
            model = load_model(os.path.join(MODEL_PATH, f'{key}_model_{i}.h5'))
            trained_model[key].append(model)


model_predictions = test_model(trained_model, datasets)