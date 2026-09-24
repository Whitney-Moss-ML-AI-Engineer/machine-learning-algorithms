"""Regression examples: Linear, Ridge, and LASSO with cross-validation."""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "linear": LinearRegression(),
    "ridge": RidgeCV(alphas=np.logspace(-3, 3, 20)),
    "lasso": LassoCV(alphas=np.logspace(-3, 1, 30), max_iter=10000, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(name, {"MAE": mean_absolute_error(y_test, pred), "RMSE": rmse, "R2": r2_score(y_test, pred)})
