import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Feature engineering

# Rolling averages of last n years

# Years in league
# for each unique player id, get the number of years they have been in the league
# Years in league

def nanHandle(df):
    """Replace infinities with NaNs and NaNs with zero."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.nan if np.isinf(x) else x)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def featureEngineering(df):
    # Calculate the years played
    df['years_played'] = df.groupby('player_id')['season'].rank(method='dense') - 1
    df['years_played'] = df['years_played'].astype(int)

    # Sort by player_id and season to ensure correct order for rolling calculations
    df = df.sort_values(by=['player_id', 'season', 'week']).reset_index(drop=True)

    # Define rolling calculations
    def calculate_rolling_features(group):
        group['rolling_passing_yards'] = group['passing_yards'].cumsum().shift(1)
        group['rolling_fantasy_points'] = group['fantasy_points'].expanding().mean().shift(1)
        group['rolling_avg_fantasy_points'] = group['fantasy_points'].rolling(window=4, min_periods=1).mean().shift(1)
        group['fp_rolling_std_4w'] = group['fantasy_points'].rolling(window=4, min_periods=1).std().shift(1)
        group['lag_fantasy_points'] = group['fantasy_points'].shift(1)
        group['moving_avg_fp_3w'] = group['fantasy_points'].rolling(window=3, min_periods=1).mean().shift(1)
        group['fp_diff_wow'] = group['fantasy_points'].diff().shift(1)
        group['fp_pct_change_2w'] = group['fantasy_points'].pct_change(periods=2).shift(1)
        group['fp_momentum_3w'] = group['fp_diff_wow'].rolling(window=3, min_periods=1).sum().shift(1)
        group['fp_acceleration'] = group['fp_diff_wow'].diff().shift(1)
        return group

    # Apply the rolling calculations within each player and season group
    df = df.groupby(['player_id', 'season']).apply(calculate_rolling_features)

    # Clean data to handle infinities
    df = nanHandle(df)

    # List of new columns created
    new_columns = [
        'years_played', 'rolling_passing_yards', 'rolling_fantasy_points',
        'rolling_avg_fantasy_points', 'fp_rolling_std_4w', 'lag_fantasy_points',
        'moving_avg_fp_3w', 'fp_diff_wow', 'fp_pct_change_2w', 'fp_momentum_3w',
        'fp_acceleration'
    ]

    # Replace NaNs with zero for the specified columns
    for column in new_columns:
        if column in df.columns:
            df[column] = df[column].fillna(0)

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

def sequence_generator(features, target, sequence_length):
    for i in range(len(features) - sequence_length):
        yield features[i:i + sequence_length], target[i + sequence_length]


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
        
        yield X_train_normalized, X_test_normalized, y_train, y_test
        
        