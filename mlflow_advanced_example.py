import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():
    
    #Enable autologging
    mlflow.sklearn.autolog()

    # Model and hyperparameters
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions and metrics
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 2)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "random_forest_model")

    print(f"Logged accuracy: {acc}")

    # Use tags to label/group runs
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("stage", "baseline")
    mlflow.set_tag("notes", "First try with default parameters")


    # Log Artifacts
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

for depth in [2, 4, 6]:
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, max_depth=depth)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("accuracy", acc)

