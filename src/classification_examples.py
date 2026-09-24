"""Classification example using Logistic Regression and Random Forest."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}\n", classification_report(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, prob))
