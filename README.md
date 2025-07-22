# MLflow Wine Classifier

This is a hands-on machine learning project demonstrating how to use **MLflow Tracking** to log experiments, parameters, metrics, and model artifacts.

We train multiple `RandomForestClassifier` models on the **Wine dataset** from scikit-learn, experimenting with different hyperparameters and logging all results using MLflow.

---

## Features Tracked with MLflow

- Model parameters (e.g. `max_depth`)
- Accuracy metric
- Confusion matrix image (as artifact)
- Saved model (sklearn)
- Tags and metadata
- Easy comparison of runs using the MLflow UI

---

## Project Structure
mlflow/
├── mini-mlflow-tracking-project # Main script with experiment logging
├── mlflow-basic-example #Basic mlflow commands
├── mlflow-advanced-example #Advanced mlflow commands
├── cm_depth2.png # Sample confusion matrix
├── cm_depth4.png
├── cm_depth6.png
├── confusion_matrix.png # From advanced example
└── README.md

## Key Learnings
- MLflow makes it easy to track and compare machine learning experiments.
- Logging artifacts like plots improves reproducibility and debugging.
- Autologging can simplify experiment tracking in larger pipelines.
