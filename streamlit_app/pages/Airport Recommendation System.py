# streamlit_app/pages/Airport Recommendation System.py
# Transparent Airport Recommender with uploads, validation, fixed-size annotated visuals

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

st.set_page_config(page_title="🧭 Airport Recommendation System", layout="wide")
st.title("🧭 Airport Recommendation System (Demo)")

st.markdown("""
**What this page does**  
- Upload **TRAIN** interactions (user_id, airport_id[, value][, ts]) and optional **VALIDATION** interactions.  
- Builds a simple, transparent **hybrid recommender**: popularity + item–item similarity.  
- Reports **HR@K / MAP@K / NDCG@K** and shows annotated visuals explaining *why* results look the way they do.  
- Optional metadata (airport_id → name) for nicer labels.
""")

# -----------------------
# Controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    topk = st.slider("Top-K for evaluation", 5, 50, 10, 1)
    alpha_pop = st.slider("Blend: popularity vs similarity", 0.0, 1.0, 0.3, 0.05,
                          help="0 = pure similarity; 1 = pure popularity")
    min_user_interactions = st.slider("Min interactions per user (train)", 1, 10, 1, 1)
    use_validation = st.checkbox("Use uploaded VALIDATION (else holdout from TRAIN)", value=True)
    st.caption("Fixed figsize + tight layout for clean screenshots.")

# -----------------------
# Uploaders
# -----------------------
train_file = st.file_uploader("Upload TRAIN interactions CSV", type=["csv"])
val_file = st.file_uploader("Optional: Upload VALIDATION interactions CSV", type=["csv"])
meta_file = st.file_uploader("Optional: Upload Airports metadata CSV", type=["csv"])

# -----------------------
# Helpers
# -----------------------
def _compact_show(fig, width=6.8, height=3.8):
    fig.set_size_inches(width, height)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False, clear_figure=True)

def _read_interactions(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if "user_id" not in df.columns or "airport_id" not in df.columns:
        raise ValueError("CSV must include user_id and airport_id columns.")
    keep = ["user_id", "airport_id"]
    if "value" in df.columns: keep.append("value")
    if "ts" in df.columns: keep.append("ts")
    df = df[keep].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["airport_id"] = df["airport_id"].astype(str)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(1.0)
    else:
        df["value"] = 1.0
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df

def _read_meta(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if "airport_id" not in df.columns:
        raise ValueError("Metadata CSV must include airport_id.")
    return df

def _per_user_holdout(df: pd.DataFrame, n_holdout=1, seed=42):
    rng = np.random.default_rng(seed)
    df = df.sort_values("user_id")
    val_rows, train_rows = [], []
    for user, grp in df.groupby("user_id"):
        if len(grp) >= n_holdout + 1:
            take = min(n_holdout, len(grp))
            idx = rng.choice(grp.index.values, size=take, replace=False)
            val_rows.append(grp.loc[idx])
            train_rows.append(grp.drop(idx))
        else:
            train_rows.append(grp)
    val_df = pd.concat(val_rows) if val_rows else df.iloc[0:0]
    train_df = pd.concat(train_rows)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def _build_user_item(df: pd.DataFrame):
    by_user, by_item = defaultdict(set), defaultdict(set)
    for _, row in df.iterrows():
        if float(row["value"]) > 0:
            u, i = row["user_id"], row["airport_id"]
            by_user[u].add(i)
            by_item[i].add(u)
    return by_user, by_item

def _item_popularity(by_item):
    return {i: len(users) for i, users in by_item.items()}

def _item_cosine_scores(by_user, by_item):
    items = list(by_item.keys())
    scores = defaultdict(dict)
    for a_idx, a in enumerate(items):
        Ua = by_item[a]
        if not Ua: 
            continue
        for b in items[a_idx + 1:]:
            Ub = by_item[b]
            if not Ub: 
                continue
            inter = len(Ua & Ub)
            if inter == 0:
                continue
            sim = inter / np.sqrt(len(Ua) * len(Ub))
            scores[a][b] = sim
            scores[b][a] = sim
    return scores

def _recommend_for_user(u, by_user, by_item, pop, sim_scores, alpha_pop=0.3, exclude_seen=True, topk=10):
    seen = by_user.get(u, set())
    candidate_items = set(by_item.keys())
    if exclude_seen:
        candidate_items -= seen
    if not candidate_items:
        return []
    out = []
    for i in candidate_items:
        # similarity term: max similarity to items the user has seen
        s = max((sim_scores.get(i, {}).get(j, 0.0) for j in seen), default=0.0) if seen else 0.0
        p = pop.get(i, 0.0)
        score = alpha_pop * p + (1.0 - alpha_pop) * s
        out.append((i, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in out[:topk]]

def _hr_at_k(gt, recs):
    return 1.0 if any(i in recs for i in gt) else 0.0

def _ap_at_k(gt, recs):
    hits, s = 0, 0.0
    for idx, r in enumerate(recs, start=1):
        if r in gt:
            hits += 1
            s += hits / idx
    return s / max(1, min(len(gt), len(recs)))

def _ndcg_at_k(gt, recs):
    def dcg(items):
        return sum((1.0 / np.log2(i + 1)) for i, r in enumerate(items, start=1) if r in gt)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt), len(recs)) + 1))
    return (dcg(recs) / ideal) if ideal > 0 else 0.0

# -----------------------
# Main
# -----------------------
if train_file is None:
    st.info("Upload TRAIN interactions to begin.")
    st.stop()

try:
    train_df = _read_interactions(train_file)
    st.success(f"TRAIN loaded: {len(train_df):,} rows, {train_df['user_id'].nunique():,} users, {train_df['airport_id'].nunique():,} airports")
except Exception as e:
    st.error(f"Failed to read TRAIN: {e}")
    st.stop()

meta_df = None
if meta_file is not None:
    try:
        meta_df = _read_meta(meta_file)
        st.success(f"Metadata loaded: {len(meta_df):,} rows")
    except Exception as e:
        st.warning(f"Metadata skipped: {e}")

# Filter users with few interactions
user_counts = train_df["user_id"].value_counts()
keep_users = set(user_counts[user_counts >= min_user_interactions].index)
train_df = train_df[train_df["user_id"].isin(keep_users)].reset_index(drop=True)

# Validation choice
if use_validation and val_file is not None:
    try:
        val_df = _read_interactions(val_file)
        val_df = val_df[val_df["user_id"].isin(train_df["user_id"].unique())].reset_index(drop=True)
        st.info("Using uploaded VALIDATION interactions.")
    except Exception as e:
        st.error(f"Failed to read VALIDATION: {e}")
        st.stop()
else:
    st.info("No VALIDATION uploaded (or toggle off). Using 1 holdout interaction per user from TRAIN.")
    train_df, val_df = _per_user_holdout(train_df, n_holdout=1, seed=42)

# Build structures
by_user_train, by_item_train = _build_user_item(train_df)
pop = _item_popularity(by_item_train)
sim_scores = _item_cosine_scores(by_user_train, by_item_train)

# Evaluate
users_eval = sorted(val_df["user_id"].unique())
if not users_eval:
    st.error("No overlapping users in VALIDATION to evaluate. Add validation rows or disable validation.")
    st.stop()

hits, maps, ndcgs = [], [], []
for u in users_eval:
    gt = set(val_df[val_df["user_id"] == u]["airport_id"].tolist())
    recs = _recommend_for_user(u, by_user_train, by_item_train, pop, sim_scores,
                               alpha_pop=alpha_pop, exclude_seen=True, topk=topk)
    hits.append(_hr_at_k(gt, recs))
    maps.append(_ap_at_k(gt, recs))
    ndcgs.append(_ndcg_at_k(gt, recs))

hr = float(np.mean(hits)) if hits else 0.0
mapk = float(np.mean(maps)) if maps else 0.0
ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

# ---------- Metrics Row ----------
c1, c2, c3 = st.columns(3)
c1.metric(f"HR@{topk}", f"{hr*100:,.1f}%")
c2.metric(f"MAP@{topk}", f"{mapk*100:,.1f}%")
c3.metric(f"NDCG@{topk}", f"{ndcg*100:,.1f}%")

# ---------- Split Overview ----------
st.subheader("Train / Validation Overview")
rows = [
    ("Train", train_df["user_id"].nunique(), train_df["airport_id"].nunique(), len(train_df)),
    ("Validation", val_df["user_id"].nunique(), val_df["airport_id"].nunique(), len(val_df)),
]
split_df = pd.DataFrame(rows, columns=["Split", "Users", "Airports", "Rows"])
st.dataframe(split_df, use_container_width=True, hide_index=True)

# ---------- Visuals ----------
# 1) Airport popularity (ANNOTATED)
st.subheader("Airport Popularity (Train)")
pop_series = pd.Series(pop, name="count").sort_values(ascending=False)
topN = min(20, len(pop_series))
labels = list(pop_series.index[:topN])
if meta_df is not None and "airport_id" in meta_df.columns and "name" in meta_df.columns:
    name_map = dict(zip(meta_df["airport_id"].astype(str), meta_df["name"].astype(str)))
    labels = [name_map.get(i, i) for i in labels]

fig1, ax1 = plt.subplots(figsize=(7.0, 3.8))
ax1.bar(range(topN), pop_series.values[:topN])
ax1.set_xticks(range(topN))
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Users (count)")
ax1.set_title("Top Airports by Unique Users (Train)", fontsize=12)
# Annotation
ax1.annotate(
    "Popularity baseline:\n• Tall bars = safe default recs\n• But can reduce personalization",
    xy=(0.01, 0.98), xycoords="axes fraction", va="top", ha="left",
    fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9)
)
_compact_show(fig1)

# 2) Blend sensitivity curve (ANNOTATED)
st.subheader("Sensitivity: HR@K vs Popularity Blend")
alphas = np.linspace(0.0, 1.0, 11)
hrs = []
for a in alphas:
    h = []
    for u in users_eval:
        gt = set(val_df[val_df["user_id"] == u]["airport_id"].tolist())
        recs = _recommend_for_user(u, by_user_train, by_item_train, pop, sim_scores,
                                   alpha_pop=a, exclude_seen=True, topk=topk)
        h.append(_hr_at_k(gt, recs))
    hrs.append(np.mean(h) if h else 0.0)

fig2, ax2 = plt.subplots(figsize=(6.8, 3.8))
ax2.plot(alphas, hrs, marker="o")
ax2.set_xlabel("alpha_pop (0 = similarity, 1 = popularity)")
ax2.set_ylabel(f"HR@{topk}")
ax2.set_title("Blend Sensitivity", fontsize=12)
ax2.set_ylim(0, 1)
# Annotation
ax2.annotate(
    "Trade-off:\n• Left side → more personalized (similarity)\n• Right side → safer, popular picks",
    xy=(0.02, 0.95), xycoords="axes fraction", va="top", ha="left",
    fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9)
)
_compact_show(fig2)

# 3) Recommendation coverage (ANNOTATED)
st.subheader("Recommendation Coverage")
all_recs = []
for u in users_eval:
    recs = _recommend_for_user(u, by_user_train, by_item_train, pop, sim_scores,
                               alpha_pop=alpha_pop, exclude_seen=True, topk=topk)
    all_recs.extend(recs)
unique_recs = len(set(all_recs))

fig3, ax3 = plt.subplots(figsize=(6.8, 3.6))
ax3.bar(["Unique Airports Recommended"], [unique_recs])
ax3.set_ylim(0, max(1, unique_recs))
ax3.set_title("Catalog Coverage of Top-K Recommendations", fontsize=12)
# Annotation
ax3.annotate(
    "Coverage measures variety:\n• Higher = we surface more of the catalog\n• Too low = same few airports for everyone",
    xy=(0.02, 0.9), xycoords="axes fraction", va="top", ha="left",
    fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9)
)
_compact_show(fig3)

# ---------- Debug / Preview ----------
with st.expander("Preview: Sample Recommendations"):
    example_users = users_eval[:5]
    rows = []
    for u in example_users:
        recs = _recommend_for_user(u, by_user_train, by_item_train, pop, sim_scores,
                                   alpha_pop=alpha_pop, exclude_seen=True, topk=topk)
        rows.append({"user_id": u, "recommendations": ", ".join(recs)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
