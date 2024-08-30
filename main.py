# Importing libraries
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit

from scripts.dataPreprocessor import dataPreprocessor
from scripts.training import train_model
from scripts.evaluation import modelEvaluation

# Initialize classes

# Initialize paths
data_path = 'data/player_stats.csv'
results_path = 'results/'
model_path = 'models/'
predictions_path = os.path.join(results_path, 'predictions_2024.xlsx')

# Load data
df = pd.read_csv(data_path)
df = df.sort_values(by=['player_id', 'season', 'week'])

dpp = dataPreprocessor(df)

# Data processing
sequence_length = 3
batch_size = 64
tscv = TimeSeriesSplit(n_splits=5)

df = dpp.feature_engineering()

position_dfs = dpp.clean_data()

qb_df_features, qb_df_target = position_dfs['QB']
rb_df_features, rb_df_target = position_dfs['RB']
wr_df_features, wr_df_target = position_dfs['WR']
te_df_features, te_df_target = position_dfs['TE']

qb_gen_batch = dpp.batch_sequence_generator(qb_df_features.values, qb_df_target.values, sequence_length, batch_size)
rb_gen_batch = dpp.batch_sequence_generator(rb_df_features.values, rb_df_target.values, sequence_length, batch_size)
wr_gen_batch = dpp.batch_sequence_generator(wr_df_features.values, wr_df_target.values, sequence_length, batch_size)
te_gen_batch = dpp.batch_sequence_generator(te_df_features.values, te_df_target.values, sequence_length, batch_size)


X_qb_list = []
y_qb_list = []
X_rb_list = []
y_rb_list = []
X_wr_list = []
y_wr_list = []
X_te_list = []
y_te_list = []

# Iterate over generator and accumulate data
for X_qb_batch, y_qb_batch in qb_gen_batch:
    X_qb_list.append(X_qb_batch)
    y_qb_list.append(y_qb_batch)

for X_rb_batch, y_rb_batch in rb_gen_batch:
    X_rb_list.append(X_rb_batch)
    y_rb_list.append(y_rb_batch)

for X_wr_batch, y_wr_batch in wr_gen_batch:
    X_wr_list.append(X_wr_batch)
    y_wr_list.append(y_wr_batch)

for X_te_batch, y_te_batch in te_gen_batch:
    X_te_list.append(X_te_batch)
    y_te_list.append(y_te_batch)


# Convert lists to single NumPy arrays
X_qb = np.concatenate(X_qb_list, axis=0)
y_qb = np.concatenate(y_qb_list, axis=0)
X_rb = np.concatenate(X_rb_list, axis=0)
y_rb = np.concatenate(y_rb_list, axis=0)
X_wr = np.concatenate(X_wr_list, axis=0)
y_wr = np.concatenate(y_wr_list, axis=0)
X_te = np.concatenate(X_te_list, axis=0)
y_te = np.concatenate(y_te_list, axis=0)


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
        

# Train the model
trained_model, model_predictions = train_model(datasets)

#eval = modelEvaluation(trained_model, datasets)
