
from tensorflow.keras.callbacks import EarlyStopping
from scripts.modelDesign import build_model_gru

def train_model(datasets, patience=50, restore_best_weights=True):

    early_stopping = EarlyStopping(
        monitor='val_loss',        # Metric to monitor
        patience=50,                # Number of epochs with no improvement to wait before stopping
        restore_best_weights=True  # Restore model weights from the epoch with the best metric value
    )

    trained_models = {key: [] for key in datasets.keys()}
    model_predictions = {key: [] for key in datasets.keys()}

    # run the model on the normalized datasets

    for key, datasets in datasets.items():
        for i, dataset in enumerate(datasets):
            print(f"Training RNN model for {key} - split {i + 1}...")
            
            # Get the input shape
            input_shape = dataset['X_train'].shape[1:]
            
            # Create the RNN model
            model = build_model_gru(input_shape, units=512)
            
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
            trained_models[key].append(model)


    # Testing
    
    for key, models in trained_models.items():
        for i, model in enumerate(models):
            dataset = datasets[key][i]
            print(f"Testing RNN model for {key} - split {i + 1}...")
            
            # Make predictions
            y_pred = model.predict(dataset['X_test'])
            
            # Store the predictions
            model_predictions[key].append(y_pred)

    # Output the predictions
    for key, predictions in model_predictions.items():
        print(f"Predictions for {key}:")
        for i, prediction in enumerate(predictions):
            print(f"  Split {i + 1}:")
            print(f"    Shape: {prediction.shape}")
            
    return trained_models, model_predictions