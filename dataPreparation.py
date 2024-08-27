import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns




# Feature engineering

# Rolling averages of last n years

# Years in league
# for each unique player id, get the number of years they have been in the league
# Years in league

def featureEngineering(df):
    df['years_played'] = df.groupby('player_id')['season'].rank(method='dense')-1
    df['years_played'] = df['years_played'].astype(int)
    
    return df



def cleanData(df):
    # Drop any rows with season="POST"
    df = df[df['season_type'] != 'POST']

    # Full df cleaning
    df.drop(
        columns=[
            'headshot_url',
            'player_name',
            'player_display_name',
            'recent_team',
            'opponent_team',
            'position',
            'season_type'],
        inplace=True
    )
    target = 'fantasy_points_ppr'

    df = pd.get_dummies(
        df,
        columns=['player_id','season']
    )

    # QB data cleaning
    qb_df = df[df['position_group'] == 'QB'].copy()

    qb_df = pd.get_dummies(
        qb_df,
        columns=['position_group']
    )


    qb_df.drop(
        columns=[
            'receiving_epa',
            'racr',
            'target_share',
            'air_yards_share',
            'wopr'
        ], inplace=True
    )
    qb_df.dropna(
        subset=[
            'passing_epa',
            'pacr',
            'dakota',
            'rushing_epa'
        ], inplace=True
    )


    qb_df_target = qb_df[target]
    qb_df_features = qb_df.drop(columns=[target])




    # RB data cleaning

    rb_df = df[df['position_group'] == 'RB'].copy()

    rb_df = pd.get_dummies(
        rb_df,
        columns=['position_group']
    )

    rb_df.drop(
        columns=[
            'passing_epa',
            'pacr',
            'dakota',
            'receiving_epa',
            'rushing_epa',
            'racr',
            'target_share',
            'air_yards_share',
            'wopr'
        ], inplace=True
    )

    rb_df_target = rb_df[target]
    rb_df_features = rb_df.drop(columns=[target])



    # WR data cleaning
    wr_df = df[df['position_group'] == 'WR'].copy()

    wr_df = pd.get_dummies(
        wr_df,
        columns=['position_group']
    )

    wr_df.drop(
        columns=[
            'passing_epa',
            'pacr',
            'dakota',
            'rushing_epa',
        ], inplace=True
    )

    wr_df.dropna(
        subset=[
            'receiving_epa',
            'racr',
            'target_share',
            'air_yards_share',
            'wopr'
        ], inplace=True
    )


    wr_df_target = wr_df[target]
    wr_df_features = wr_df.drop(columns=[target])



    # TE data cleaning
    te_df = df[df['position_group'] == 'TE'].copy()

    te_df = pd.get_dummies(
        te_df,
        columns=['position_group']
    )

    te_df.drop(
        columns=[
            'passing_epa',
            'pacr',
            'dakota',
            'rushing_epa',
        ], inplace=True
    )

    te_df.dropna(
        subset=[
            'receiving_epa',
            'racr',
            'target_share',
            'air_yards_share',
            'wopr'
        ], inplace=True
    )

    te_df_target = te_df[target]
    te_df_features = te_df.drop(columns=[target])



    #sequence_length = 4  # Example: 4 weeks of data
    #X_qb, y_qb = create_sequences(np.hstack((qb_features, qb_target.reshape(-1, 1))), sequence_length)
    
    return qb_df_features, qb_df_target, rb_df_features, rb_df_target, wr_df_features, wr_df_target, te_df_features, te_df_target

def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(target[i+sequence_length])
        
        
    return np.array(X), np.array(y)

def normalize_sequences(X_train, X_test):

    scaler = StandardScaler()
    num_features = X_train.shape[2]  # Number of features
    
    # Reshape to 2D for normalization
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_test_reshaped = X_test.reshape(-1, num_features)
    
    # Fit scaler on the training data only
    scaler.fit(X_train_reshaped)
    
    # Transform both training and test data
    X_train_normalized = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_normalized = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    return X_train_normalized, X_test_normalized

def process_data(X, y, tscv):
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Normalize the sequences
        X_train_normalized, X_test_normalized = normalize_sequences(X_train, X_test)
        
    return X_train_normalized, X_test_normalized, y_train, y_test
        
        