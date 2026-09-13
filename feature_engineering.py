import pandas as pd
import numpy as np
import re, json

df = pd.read_csv('/mnt/user-data/uploads/BoxRec_Boxers_Data.csv')

# --- Parse KO% ---
def parse_pct(x):
    if pd.isna(x): return np.nan
    return float(str(x).replace('%',''))
df['ko_pct'] = df['KOs'].apply(parse_pct)

# --- Parse Career length ---
def parse_career(x):
    if pd.isna(x): return np.nan
    parts = str(x).split('-')
    if len(parts) != 2: return np.nan
    try:
        return int(parts[1]) - int(parts[0])
    except: return np.nan
df['career_years'] = df['Career'].apply(parse_career)
df['career_years'] = df['career_years'].replace(0,1)  # debut year same

# --- Parse height/reach to inches ---
def parse_inches(x):
    if pd.isna(x): return np.nan
    m = re.search(r'([\d.]+)cm', str(x))
    if m: return float(m.group(1)) / 2.54
    return np.nan
df['height_in'] = df['Height'].apply(parse_inches)
df['reach_in'] = df['Reach'].apply(parse_inches)

# --- Derived stats ---
df['bouts'] = df['bouts'].astype(float)
df['rounds'] = df['rounds'].astype(float)
df['rounds_per_bout'] = df['rounds'] / df['bouts']
df['bouts_per_year'] = df['bouts'] / df['career_years']
df['win_rate'] = df['Win'] / df['bouts']
df['stoppage_wins_est'] = df['ko_pct'] / 100.0 * df['Win']  # rough

# --- Impute missing numeric features (median, by division where sensible) ---
for col in ['reach_in','height_in']:
    df[col] = df.groupby('division')[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df[col].fillna(df[col].median())

df['ko_pct'] = df['ko_pct'].fillna(df['ko_pct'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['career_years'] = df['career_years'].fillna(df['career_years'].median())
df['rounds_per_bout'] = df['rounds_per_bout'].fillna(df['rounds_per_bout'].median())
df['bouts_per_year'] = df['bouts_per_year'].replace([np.inf,-np.inf], np.nan)
df['bouts_per_year'] = df['bouts_per_year'].fillna(df['bouts_per_year'].median())

df['reach_height_ratio'] = df['reach_in'] / df['height_in']
df['stance_southpaw'] = (df['Stance'] == 'southpaw').astype(int)

# division weight order (lightest to heaviest), used as ordinal numeric feature
division_order = ['atom','minimum','light fly','fly','super fly','bantam','super bantam',
                   'feather','super feather','light','super light','welter','super welter',
                   'middle','super middle','light heavy','cruiser','heavy']
div_map = {d:i for i,d in enumerate(division_order)}
df['division_ord'] = df['division'].map(div_map)

# titles count (rough signal of accomplishment, separate from clustering features but useful display)
df['has_title'] = df['titles'].notna().astype(int)

print(df[['Name','ko_pct','career_years','rounds_per_bout','bouts_per_year','win_rate',
          'height_in','reach_in','Age','division_ord','stance_southpaw']].head(10))
print(df.shape)
df.to_pickle('/home/claude/boxing/df_engineered.pkl')
