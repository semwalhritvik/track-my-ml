import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
X,y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=26)

for depth in [2,4,6]:
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, max_depth=depth)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        #Log
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "rf_model")

        #Save and log artifact
        disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
        plot_file = f"cm_depth{depth}.png"
        plt.savefig(plot_file)
        mlflow.log_artifact(plot_file)
        
        mlflow.set_tag("dataset", "wine")
        mlflow.set_tag("purpose", "depth tuning")

