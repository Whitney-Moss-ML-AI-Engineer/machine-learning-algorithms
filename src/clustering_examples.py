"""K-Means clustering with standardized features."""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = load_wine(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

for k in range(2, 7):
    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = model.fit_predict(X_scaled)
    print(f"k={k}: inertia={model.inertia_:.2f}, silhouette={silhouette_score(X_scaled, labels):.3f}")
