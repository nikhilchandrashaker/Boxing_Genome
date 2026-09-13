import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_pickle('/home/claude/boxing/df_engineered.pkl')

feature_cols = ['ko_pct','career_years','rounds_per_bout','bouts_per_year','win_rate',
                'height_in','reach_in','reach_height_ratio','Age']

X = df[feature_cols].copy()
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# find best k via silhouette, preferring balanced clusters (avoid degenerate tiny clusters)
best_k, best_score = None, -1
scores = {}
for k in range(3,9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    s = silhouette_score(Xs, labels)
    min_cluster_size = pd.Series(labels).value_counts().min()
    scores[k] = (s, min_cluster_size)
print("silhouette scores (score, min_cluster_size):", scores)
best_k = 5  # chosen: balanced cluster sizes (43-94), silhouette 0.147 (close to global max 0.156 at k=6,
            # but k=6 produces a degenerate 7-member outlier cluster driven by one noisy feature ratio)
best_score = scores[best_k][0]
print("chosen k:", best_k, best_score)

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(Xs)

# PCA to 2D for the "universe"
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xs)
df['pca_x'] = coords[:,0]
df['pca_y'] = coords[:,1]
print("explained var:", pca.explained_variance_ratio_)

# cluster centroid profile relative to population mean (z-scored), to auto-derive descriptive labels
overall_mean = X.mean()
overall_std = X.std()
centroid_z = {}
for c in range(best_k):
    sub = X[df['cluster']==c]
    z = (sub.mean() - overall_mean) / overall_std
    centroid_z[c] = z.sort_values(ascending=False)
    print(f"\n--- Cluster {c} (n={len(sub)}) top distinguishing features ---")
    print(z.sort_values(ascending=False))

df.to_pickle('/home/claude/boxing/df_clustered.pkl')
import pickle
with open('/home/claude/boxing/scaler_pca.pkl','wb') as f:
    pickle.dump({'scaler':scaler,'pca':pca,'feature_cols':feature_cols}, f)
