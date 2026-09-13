# The Boxing Genome

Unsupervised discovery of fighter archetypes from BoxRec career statistics, plus a fighter-similarity search and an interactive "tale of the tape" explorer.

Instead of predicting fight outcomes, this project asks: **what makes a boxer who they are, and are there natural clusters of fighting style hidden in the stats?**

## What's in this repo

| File | What it is |
|---|---|
| `BoxRec_Boxers_Data.csv` | Source data — 328 professional boxers scraped from BoxRec |
| `feature_engineering.py` | Cleans raw fields, engineers style features |
| `clustering.py` | Standardizes features, runs k-means + PCA |
| `export.py` | Computes similarity scores, DNA percentiles, writes final JSON |
| `boxing_genome_data.json` | Final dataset consumed by the UI |
| `BoxingGenome.jsx` | Interactive React artifact (search, DNA radar, similarity, universe map) |

## Data

328 boxers, one row per fighter, with career totals (not fight-by-fight history): division, rating, bouts, rounds, KO%, career span, physical measurements, stance, nationality, W-L-D record, titles.

## Method

**1. Feature engineering** — from the raw fields, derived:
- KO rate (parsed from the `KOs` percentage string)
- Career length in years (from `Career` span)
- Rounds per bout (a proxy for going the distance vs. finishing)
- Bouts per year (activity level)
- Win rate
- Height and reach in inches (parsed from the raw `5′ 8″ / 173cm` format)
- Reach-to-height ratio

Missing reach/height (~37%/5% of rows) were imputed with the division median, then the overall median as fallback. Fields that were essentially unusable — `vada` (100% missing) and `titles` (80% missing, kept only as a display field, not a clustering input) — were excluded from the model.

**2. Clustering** — 9 features, standardized, k-means tested for k = 3–8 using silhouette score. k=6 scored marginally higher (0.156 vs 0.147) but produced a degenerate 7-member cluster driven by one noisy ratio. **k=5** was chosen instead: balanced group sizes (43–94 fighters) at a very similar silhouette score.

**3. Archetypes** — each cluster was named by inspecting which features were most above/below the population average for that group (not chosen in advance):

- **The Rising Destroyer** (43) — young, high-volume, high-KO%, short career so far
- **The Iron Veteran** (60) — long career, older, goes deep into fights, lower power
- **The Compact Finisher** (87) — smaller, less tenured, converts fights into quick finishes
- **The Technician** (94, largest group) — goes the distance far more than they stop opponents
- **The Long-Range Operator** (44) — tall, long reach, experienced, strong win rate and power

**4. Boxing DNA** — six percentile scores per fighter (Power, Technique, Volume, Longevity, Reach, Win rate), rendered as a radar/hexagon.

**5. Similarity search** — Euclidean distance between fighters in the standardized 9-feature space, converted to a 0–100 similarity score. Top 8 matches are precomputed for every fighter.

**6. Universe map** — PCA to 2 components (explains ~50% of variance) for a browsable 2D layout where nearby points share a statistical style.

## Honest limitations

- **Silhouette score is ~0.15** — the clusters are real but soft and overlapping, not five hard, cleanly separated categories. That's a believable finding for boxing, which has genuine hybrid styles, but it means the archetype labels are directional, not definitive.
- **No trajectory analysis.** This dataset is one row per fighter — current career totals — not a fight-by-fight time series. It cannot show how an individual fighter's style changed over their career (e.g. "does KO% decline with age"). That would require per-fight or per-year data, which isn't in this file. The original project brainstorm included a "Boxer Evolution" feature; it was deliberately left out here rather than faked.
- **328 fighters is a small sample** for unsupervised learning — fine for a portfolio-scale exploration, but not grounds for claiming statistically robust taxonomy of professional boxing at large.
- **`rating`** was excluded from clustering — the vast majority of fighters in this dataset are rated 5.0, so it carries almost no discriminating signal.
- Reach/height imputation and the PCA projection are approximations; treat exact similarity percentages and archetype boundaries as illustrative, not precise measurements.

## Reproducing

```bash
pip install pandas numpy scikit-learn scipy
python3 feature_engineering.py   # -> df_engineered.pkl
python3 clustering.py            # -> df_clustered.pkl, scaler_pca.pkl
python3 export.py                # -> boxing_genome_data.json
```

Then open `BoxingGenome.jsx` as a React artifact — the dataset is embedded directly in the file, no build step or server required.
