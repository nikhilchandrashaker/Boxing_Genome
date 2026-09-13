import pandas as pd, numpy as np, json, pickle
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_pickle('/home/claude/boxing/df_clustered.pkl')
with open('/home/claude/boxing/scaler_pca.pkl','rb') as f:
    meta = pickle.load(f)
scaler, feature_cols = meta['scaler'], meta['feature_cols']

X = df[feature_cols].copy()
Xs = scaler.transform(X)

# pairwise distances -> similarity 0-100
D = euclidean_distances(Xs)
maxD = D.max()
SIM = 100 * (1 - D / maxD)

archetypes = {
    0: {"name": "The Rising Destroyer",
        "blurb": "Young, active, and heavy-handed — high knockout rate and a busy fight schedule early in a still-short career, before facing the test of championship rounds."},
    1: {"name": "The Iron Veteran",
        "blurb": "Long career, older, and comfortable going deep into fights — but lower activity and a lower knockout rate, the profile of a durable, experienced hand who wins on volume and grit rather than power."},
    2: {"name": "The Compact Finisher",
        "blurb": "Smaller and less tenured than most, but converts a good share of fights into quick finishes rather than decisions — an aggressive, early-career puncher."},
    3: {"name": "The Technician",
        "blurb": "The largest group in the dataset: fighters who go the distance far more than they stop opponents. Built on rounds, points, and craft rather than one-punch power."},
    4: {"name": "The Long-Range Operator",
        "blurb": "Tall with a long reach relative to height, experienced, and with a strong win rate and above-average power — the physically gifted, accomplished profile."},
}

from scipy.stats import rankdata
def pctile(s):
    return (rankdata(s) - 1) / (len(s) - 1) * 100

pct_power = pctile(df['ko_pct'])
pct_technique = pctile(df['rounds_per_bout'])
pct_volume = pctile(df['bouts_per_year'])
pct_longevity = pctile(df['career_years'])
pct_reach = pctile(df['reach_in'])
pct_winrate = pctile(df['win_rate'])

records = []
for i, row in df.iterrows():
    sims = SIM[i].copy()
    sims[i] = -1
    top_idx = np.argsort(sims)[::-1][:8]
    similar = [{"name": df.iloc[j]['Name'], "score": round(float(sims[j]),1), "division": df.iloc[j]['division'],
                "cluster": int(df.iloc[j]['cluster'])} for j in top_idx]

    career = row['Career']
    record = {
        "name": row['Name'],
        "division": row['division'],
        "nationality": row['Nationality'],
        "stance": row['Stance'] if pd.notna(row['Stance']) else "unknown",
        "age": float(row['Age']) if pd.notna(row['Age']) else None,
        "rating": float(row['rating']),
        "bouts": int(row['bouts']),
        "rounds": int(row['rounds']),
        "win": int(row['Win']), "lose": int(row['Lose']), "draw": int(row['Draw']),
        "career": career,
        "career_years": float(row['career_years']),
        "ko_pct": round(float(row['ko_pct']),1),
        "rounds_per_bout": round(float(row['rounds_per_bout']),2),
        "bouts_per_year": round(float(row['bouts_per_year']),2),
        "win_rate": round(float(row['win_rate'])*100,1),
        "height_in": round(float(row['height_in']),1),
        "reach_in": round(float(row['reach_in']),1),
        "has_title": bool(row['has_title']),
        "titles": row['titles'] if pd.notna(row['titles']) else None,
        "cluster": int(row['cluster']),
        "archetype": archetypes[int(row['cluster'])]['name'],
        "archetype_blurb": archetypes[int(row['cluster'])]['blurb'],
        "pca_x": round(float(row['pca_x']),3),
        "pca_y": round(float(row['pca_y']),3),
        "similar": similar,
        "dna": {
            "power": round(float(pct_power[i]),1),
            "technique": round(float(pct_technique[i]),1),
            "volume": round(float(pct_volume[i]),1),
            "longevity": round(float(pct_longevity[i]),1),
            "reach": round(float(pct_reach[i]),1),
            "win_rate": round(float(pct_winrate[i]),1),
        }
    }
    records.append(record)

cluster_summary = []
for c, info in archetypes.items():
    n = int((df['cluster']==c).sum())
    cluster_summary.append({"cluster": c, "name": info['name'], "blurb": info['blurb'], "count": n})

out = {
    "archetypes": cluster_summary,
    "fighters": records,
    "meta": {
        "n_fighters": len(records),
        "features_used": feature_cols,
        "pca_explained_variance": [0.270, 0.235],
        "silhouette_score": 0.147
    }
}

with open('/home/claude/boxing/boxing_genome_data.json','w') as f:
    json.dump(out, f)

print("Records:", len(records))
print("File size (KB):", round(len(json.dumps(out))/1024,1))
print(json.dumps(records[5], indent=2)[:1500])
