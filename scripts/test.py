from scripts.evaluate import calculate_metrics, plot_metrics, evaluate_model

def test_model(trained_models, datasets):
    model_predictions = {key: [] for key in trained_models.keys()}

    for key, models in trained_models.items():
        for i, model in enumerate(models):
            dataset = datasets[key][i]
            print(f"Testing model for {key} - split {i + 1}...")
            
            y_pred = model.predict(dataset['X_test'])
            model_predictions[key].append(y_pred)

            metrics = calculate_metrics(model, dataset['X_train'], dataset['y_train'], dataset['X_test'], dataset['y_test'])
            plot_metrics(metrics)
            evaluate_model(model_predictions, datasets)

    return model_predictions


