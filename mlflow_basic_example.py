import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():
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
