import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class dataPreprocessor:
    
    def __init__(self, df):
        self.df = df

    def nan_handle(self):
        """Replace infinities with NaNs and NaNs with zero."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].applymap(lambda x: np.nan if np.isinf(x) else x)
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)

    def feature_engineering(self):
        # Calculate the years played
        self.df['years_played'] = self.df.groupby('player_id')['season'].rank(method='dense') - 1
        self.df['years_played'] = self.df['years_played'].astype(int)

        # Sort by player_id and season to ensure correct order for rolling calculations
        self.df = self.df.sort_values(by=['player_id', 'season', 'week']).reset_index(drop=True)

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
        self.df = self.df.groupby(['player_id', 'season']).apply(calculate_rolling_features)

        # Clean data to handle infinities
        self.nan_handle()

        # List of new columns created
        new_columns = [
            'years_played', 'rolling_passing_yards', 'rolling_fantasy_points',
            'rolling_avg_fantasy_points', 'fp_rolling_std_4w', 'lag_fantasy_points',
            'moving_avg_fp_3w', 'fp_diff_wow', 'fp_pct_change_2w', 'fp_momentum_3w',
            'fp_acceleration'
        ]

        # Replace NaNs with zero for the specified columns
        for column in new_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].fillna(0)

    def clean_data(self):
        # Drop any rows with season="POST"
        self.df = self.df[self.df['season_type'] != 'POST']

        # Full df cleaning
        self.df.drop(
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

        self.df = pd.get_dummies(
            self.df,
            columns=['player_id', 'season']
        )

        position_dfs = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_df = self.df[self.df['position_group'] == position].copy()
            pos_df = pd.get_dummies(pos_df, columns=['position_group'])
            if position == 'QB':
                pos_df.drop(
                    columns=[
                        'receiving_epa', 'racr', 'target_share', 'air_yards_share', 'wopr'
                    ], inplace=True
                )
                pos_df.dropna(
                    subset=['passing_epa', 'pacr', 'dakota', 'rushing_epa'], inplace=True
                )
            else:
                pos_df.drop(
                    columns=[
                        'passing_epa', 'pacr', 'dakota', 'rushing_epa'
                    ], inplace=True
                )
                pos_df.dropna(
                    subset=['receiving_epa', 'racr', 'target_share', 'air_yards_share', 'wopr'],
                    inplace=True
                )
            pos_df_target = pos_df[target]
            pos_df_features = pos_df.drop(columns=[target])
            position_dfs[position] = (pos_df_features, pos_df_target)

        return position_dfs

    @staticmethod
    def create_sequences(features, target, sequence_length):
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        return np.array(X), np.array(y)

    @staticmethod
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

    def process_data(self, X, y, tscv):
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Normalize the sequences
            X_train_normalized, X_test_normalized = self.normalize_sequences(X_train, X_test)

            yield X_train_normalized, X_test_normalized, y_train, y_test
