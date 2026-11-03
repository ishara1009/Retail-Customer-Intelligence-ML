
"""KMeans customer segmentation for one historical snapshot."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, RANDOM_SEED, ensure_output_directories
from src.data_loader import load_monthly_data
from src.evaluation import save_json
from src.visualization import plot_clusters

CLUSTER_FEATURES = ["days_since_last_purchase", "orders_last_90d", "spend_last_90d_lkr"]


def run_clustering(data: pd.DataFrame | None = None) -> dict:
    ensure_output_directories()
    data = load_monthly_data() if data is None else data
    snapshot = data.loc[data["snapshot_date"].eq(pd.Timestamp("2025-11-01"))].copy()
    if snapshot.empty or snapshot["customer_id"].duplicated().any():
        raise ValueError("Clustering requires one customer row at snapshot 2025-11-01")
    preprocessing = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    scaled = preprocessing.fit_transform(snapshot[CLUSTER_FEATURES])
    scores = {}
    for cluster_count in [2, 3, 4, 5, 6]:
        labels = KMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init=10).fit_predict(scaled)
        scores[str(cluster_count)] = float(silhouette_score(scaled, labels))
    selected_k = int(max(scores, key=scores.get))
    model = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED, n_init=10)
    snapshot["cluster"] = model.fit_predict(scaled)

    assignments = snapshot[["customer_id", "snapshot_date", "cluster"] + CLUSTER_FEATURES]
    assignments.to_csv(PREDICTIONS_DIR / "clustering_assignments.csv", index=False)
    profiles = snapshot.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    profiles["customer_count"] = snapshot.groupby("cluster").size()
    profiles.to_csv(METRICS_DIR / "clustering_profiles.csv")
    plot_clusters(snapshot, FIGURES_DIR / "clustering_scatter.png")
    report = {
        "snapshot_date": "2025-11-01",
        "features": CLUSTER_FEATURES,
        "silhouette_scores": scores,
        "selected_k": selected_k,
        "cluster_sizes": {str(key): int(value) for key, value in snapshot["cluster"].value_counts().sort_index().items()},
        "interpretation_note": "Cluster numbers are arbitrary. Silhouette measures geometric separation, not business value.",
    }
    save_json(report, METRICS_DIR / "clustering_metrics.json")
    return report
