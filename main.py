import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from scripts.modelDesign import build_model_gru
from scripts.dataPreprocessor import dataPreprocessor
from scripts.training import train_model

data_path = 'data/player_stats.csv'
results_path = 'results/'
model_path = 'models/'
predictions_path = os.path.join(results_path, 'predictions_2024.xlsx')

df = pd.read_csv(data_path)
df = df.sort_values(by=['player_id', 'season', 'week'])

# Data processing

dpp = dataPreprocessor()

sequence_length = 3
tscv = TimeSeriesSplit(n_splits=5)

df = dpp.featureEngineering(df)

qb_df_features, qb_df_target, rb_df_features, rb_df_target, wr_df_features, wr_df_target, te_df_features, te_df_target = dpp.cleanData(df)

X_qb, y_qb = dpp.create_sequences(qb_df_features.values, qb_df_target.values, sequence_length)
X_rb, y_rb = dpp.create_sequences(rb_df_features.values, rb_df_target.values, sequence_length)
X_wr, y_wr = dpp.create_sequences(wr_df_features.values, wr_df_target.values, sequence_length)
X_te, y_te = dpp.create_sequences(te_df_features.values, te_df_target.values, sequence_length)

datasets = {'qb': [], 'rb': [], 'wr': [], 'te': []}

for X, y, key in [(X_qb, y_qb, 'qb'), 
                  (X_rb, y_rb, 'rb'), 
                  (X_wr, y_wr, 'wr'), 
                  (X_te, y_te, 'te')]:
    for i, (X_train, X_test, y_train, y_test) in enumerate(dpp.process_data(X, y, tscv)):
        datasets[key].append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        

trained_model, model_predictions = train_model(datasets)