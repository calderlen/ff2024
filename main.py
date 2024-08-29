from modelDesign import *

df = pd.read_csv('data/player_stats.csv')
df = df.sort_values(by=['player_id', 'season', 'week'])


# Data preparation

sequence_length = 3
tscv = TimeSeriesSplit(n_splits=5)

df = featureEngineering(df)

qb_df_features, qb_df_target, rb_df_features, rb_df_target, wr_df_features, wr_df_target, te_df_features, te_df_target = cleanData(df)

X_qb, y_qb = create_sequences(qb_df_features.values, qb_df_target.values, sequence_length)
X_rb, y_rb = create_sequences(rb_df_features.values, rb_df_target.values, sequence_length)
X_wr, y_wr = create_sequences(wr_df_features.values, wr_df_target.values, sequence_length)
X_te, y_te = create_sequences(te_df_features.values, te_df_target.values, sequence_length)

datasets = {'qb': [], 'rb': [], 'wr': [], 'te': []}

for X, y, key in [(X_qb, y_qb, 'qb'), 
                  (X_rb, y_rb, 'rb'), 
                  (X_wr, y_wr, 'wr'), 
                  (X_te, y_te, 'te')]:
    for i, (X_train, X_test, y_train, y_test) in enumerate(process_data(X, y, tscv)):
        datasets[key].append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        

# Training

early_stopping = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=50,                # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best metric value
)

# run the model on the normalized datasets
rnn_models_gru = {
    'qb': [],
    'rb': [],
    'wr': [],
    'te': []
}

for key, datasets in datasets.items():
    for i, dataset in enumerate(datasets):
        print(f"Training RNN model for {key} - split {i + 1}...")
        
        # Get the input shape
        input_shape = dataset['X_train'].shape[1:]
        
        # Create the RNN model
        model = buildModelGRU(input_shape, units=512)
        
        # Train the model
        model.fit(
            dataset['X_train'],
            dataset['y_train'],
            epochs=150,
            batch_size=32,
            validation_data=(dataset['X_test'], dataset['y_test']),
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # Store the model
        rnn_models_gru[key].append(model)


# Testing

rnn_predictions_gru = {'qb': [], 'rb': [], 'wr': [], 'te': []}

for key, models in rnn_models_gru.items():
    for i, model in enumerate(models):
        dataset = datasets[key][i]
        print(f"Testing RNN model for {key} - split {i + 1}...")
        
        # Make predictions
        y_pred = model.predict(dataset['X_test'])
        
        # Store the predictions
        rnn_predictions_gru[key].append(y_pred)

# Output the predictions
for key, predictions in rnn_predictions_gru.items():
    print(f"Predictions for {key}:")
    for i, prediction in enumerate(predictions):
        print(f"  Split {i + 1}:")
        print(f"    Shape: {prediction.shape}")
        