import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import sys

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# === 1. Load Data ===
df = pd.read_csv("data/spam.csv")

# === 2. Preprocess ===
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
# Add English stop words
vectorizer = CountVectorizer(stop_words='english')

X = df["text"]
y = df["label"].map({"ham": 0, "spam": 1})

X_vec = vectorizer.fit_transform(X)


# === 3. Model & Params ===
C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0  # regularization param
model = LogisticRegression(C=C)
model.fit(X_vec, y)

# === 4. Evaluation ===
y_pred = model.predict(X_vec)
acc = accuracy_score(y, y_pred)

# === 5. MLflow Logging ===
mlflow.start_run()
mlflow.log_param("C", C)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

print(f"Accuracy: {acc:.3f}")
