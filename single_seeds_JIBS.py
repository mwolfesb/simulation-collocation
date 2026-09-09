from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import shap

## Configuration
# Single Seeds for analysis
SEEDS = [(4612, "Complete Avoidance"), (7846, "Complete Collocation"), 
         (2525, "Low"), (3721, "High"), # Competitive intensity
         (6469, "Low"), (4203, "High"), # Entry barriers
         (376, "Low"), (7247, "High"), # Market size variation
         (6666, "Low"), (3650, "High")] # Market size ratio

# Define paths
BASE_PATH = "simulation"
TRACE_TEMPLATE = "{base}/{seed}/models/decision_trace.parquet"        # per-step neural network values
PRESENCE_A_TEMPLATE = "{base}/{seed}/results/firmfirm_0_p.csv"        # focal firm presence
PRESENCE_B_TEMPLATE = "{base}/{seed}/results/firmfirm_1_p.csv"        # rival firm presence
RATIO_SOURCE = "results/merged_data.csv"                             # cumulative cash-flow share
OUTPUT_DIR = Path(BASE_PATH) / "results" / "shap"                    # where figures are saved

# Define neural network / SHAP specific parameters
N_YEARS = 50                        # simulation horizon
TARGET = "chosen_action"            # policy output SHAP explains
NOOP_ACTION = 50                    # "do nothing" action index (not an entry)
MAX_ROWS_PER_FIT = 200_000          # cap rows used to train the surrogate
RANDOM_STATE = 0                    # reproducible

# Define observation channels and colors for plotting
OBS_CHANNEL_NAMES = ["own.presence", "distance", "competitor.presence", "market.size"]
PANEL_CHANNELS = ["own.presence", "competitor.presence", "market.size"] # Distance removed as static
CHANNEL_COLORS = {"own.presence": "#1f77b4", "competitor.presence": "#ff7f0e",
                  "market.size": "#2ca02c", "distance": "#7f7f7f"} 


## SHAP computation: per-entry channel contributions

# Helper functions
def obs_columns(df):
    return sorted([c for c in df.columns if c.startswith("obs_")],
                  key=lambda c: int(c.split("_")[1]))


def channel_map_for(df, obs_cols):
    # Map each obs_i column back to its channel
    c = len(obs_cols) // len(OBS_CHANNEL_NAMES)
    return {f"obs_{i * c + k}": name
            for i, name in enumerate(OBS_CHANNEL_NAMES)
            for k in range(c) if f"obs_{i * c + k}" in df.columns}


def compute_decision_channels(seed, rng):
    # One row per unique year-entry
    path = TRACE_TEMPLATE.format(base=BASE_PATH, seed=seed)
    df = pd.read_parquet(path)
    if "learning_agent" in df.columns:
        df = df[df["learning_agent"].astype(bool)]     # keep only the learning agent
    final_eps = sorted(df["episode"].unique())[-50:]   # last 50 episodes = converged policy
    df = df[df["episode"].isin(final_eps)]
    df = df.sort_values(["episode", "agent", "year"]).reset_index(drop=True)

    obs_cols = obs_columns(df)
    channel_map = channel_map_for(df, obs_cols)
    X_obs = df[obs_cols].astype("float32")

    # Surrogate fit on all kept episodes
    n_fit = min(MAX_ROWS_PER_FIT, len(df))
    fit_idx = rng.choice(len(df), size=n_fit, replace=False)
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_obs.iloc[fit_idx], df[TARGET].iloc[fit_idx])

    # Assign entries and features to correct rows/year
    prev_action = df.groupby(["episode", "agent"])[TARGET].shift(1) 
    prev_ok = df[["episode", "agent"]].eq(df[["episode", "agent"]].shift(1)).all(axis=1) 
    prev_ok.iloc[0] = False
    prev_np = prev_action.to_numpy()
    valid = (df[TARGET].ne(prev_action).to_numpy() & prev_ok.to_numpy()
             & ~prev_action.isna().to_numpy() & (prev_np != NOOP_ACTION))
    change_pos = np.flatnonzero(valid)                 # rows where an entry completes
    attr_pos = change_pos - 1 

    entered = prev_np[change_pos].astype(int)          # country entered that year
    years = df["year"].to_numpy()[change_pos]          # year the entry completes
    episode = df["episode"].to_numpy()[change_pos]

    ep = int(df["episode"].max())                      # restrict plotted entries to the last episode
    m = episode == ep
    change_pos, attr_pos, entered, years, episode = (
        change_pos[m], attr_pos[m], entered[m], years[m], episode[m])

    # Keep most recent episode
    keep = {}
    for i in range(len(change_pos)):
        k = (int(years[i]), int(entered[i]))
        if k not in keep or episode[i] > episode[keep[k]]:
            keep[k] = i
    rep = np.array(sorted(keep.values()))

    # SHAP on pre-entry rows
    shap_vals = _signed_shap(model, X_obs.iloc[attr_pos[rep]])    # SHAP on pre-entry rows
    # Pool per-country SHAP into channels
    pos = {c: i for i, c in enumerate(obs_cols)}
    chan = {}
    for ch in PANEL_CHANNELS:
        idxs = [pos[c] for c in obs_cols if channel_map.get(c) == ch]
        chan[ch] = shap_vals[:, idxs].sum(axis=1)
    dec = pd.DataFrame(chan)
    dec["year"] = years[rep]
    dec["entered_country"] = entered[rep]
    return dec.sort_values("year").reset_index(drop=True)


def _signed_shap(model, X):
    # Signed SHAP per row
    sv = shap.TreeExplainer(model).shap_values(X)
    preds = model.predict(X)
    idx = {c: i for i, c in enumerate(model.classes_)}
    # Dependence on SHAP version
    if isinstance(sv, list):                           # older SHAP
        return np.array([sv[idx[p]][i] for i, p in enumerate(preds)], dtype=np.float32)
    sv = np.asarray(sv)
    if sv.ndim == 3:                                   # newer SHAP
        return np.array([sv[i, :, idx[p]] for i, p in enumerate(preds)], dtype=np.float32)
    return sv.astype(np.float32)


## Load other datasets
# Presence (and convert into long format)
def presence_long(path):
    # Presence CSV (countries x years) to long format
    df = pd.read_csv(path).rename(columns={pd.read_csv(path, nrows=0).columns[0]: "country"})
    long = df.melt(id_vars="country", var_name="year", value_name="present")
    long["year"] = pd.to_numeric(long["year"], errors="coerce") + 1
    long["country"] = pd.to_numeric(long["country"], errors="coerce") + 1
    long["present"] = pd.to_numeric(long["present"], errors="coerce").fillna(0).astype(int)
    return long.dropna(subset=["year"])

def load_grid(seed):
    # Merge both firms' presence into one grid with A/B flags
    a = presence_long(PRESENCE_A_TEMPLATE.format(base=BASE_PATH, seed=seed))
    b = presence_long(PRESENCE_B_TEMPLATE.format(base=BASE_PATH, seed=seed))
    g = a.rename(columns={"present": "A"}).merge(
        b.rename(columns={"present": "B"}), on=["country", "year"], how="outer")
    g["A"] = g["A"].fillna(0).astype(int)
    g["B"] = g["B"].fillna(0).astype(int)
    return g

# Cumulative return ratio
def load_ratio(seed):
    # Focal firm's cumulative cash-flow share over years, for this seed
    src = Path(BASE_PATH) / RATIO_SOURCE
    sim = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    sub = sim[sim["Seed"] == seed]
    return pd.DataFrame({
        "year": pd.to_numeric(sub["year"], errors="coerce") + 1,
        "ratio": pd.to_numeric(sub["Cumu_profit_share-firm_0"], errors="coerce"),
    }).dropna().sort_values("year")



## Create the three separate panels
# Draw the presence grid panel
def draw_presence(ax, grid):
    countries = sorted(grid["country"].unique())
    ymin, ymax = min(countries), max(countries)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for c in np.arange(ymin - 0.5, ymax + 1.5):
        ax.axhline(c, color="0.88", linewidth=0.25, zorder=0)
    for y in np.arange(0.5, N_YEARS + 1.5):
        ax.axvline(y, color="0.88", linewidth=0.25, zorder=0)

    a = grid[grid["A"] == 1]
    # firm A = solid filled cells
    ax.add_collection(PatchCollection(
        [Rectangle((yr - 0.5, c - 0.5), 1.0, 1.0) for yr, c in zip(a["year"], a["country"])],
        facecolor="0.30", edgecolor="0.30", linewidth=0.4, zorder=2))
    b = grid[grid["B"] == 1]
    # firm B = X markers
    ax.scatter(b["year"], b["country"], marker="x", s=46, c="black",
               linewidth=1.6, zorder=3)

    ax.set_ylabel("Country")
    ax.set_title("Presence grid", loc="center")
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_yticks(countries)
    ax.tick_params(axis="y", labelsize=6)
    handles = [Patch(facecolor="0.30", edgecolor="0.30", label="Focal firm (A)"),
               Line2D([0], [0], marker="x", color="black", linestyle="None",
                      markersize=8, markeredgewidth=1.6, label="Rival firm (B)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=11, title="Country presence")

# Draw the cumulative return ratio panel
def draw_ratio(ax, ratio):
    ax.plot(ratio["year"], ratio["ratio"], color="black", linewidth=0.9, zorder=2)
    ax.scatter(ratio["year"], ratio["ratio"], color="black", s=14, zorder=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Return ratio (Focal firm A / Total)")
    ax.set_title("Relative return ratio over time "
                 "(Cumulative cash flow share for focal firm A)", loc="center")

# Draw the SHAP panel
def draw_shap(ax, dec):
    color_for = {c: CHANNEL_COLORS[c] for c in PANEL_CHANNELS}
    d = dec.sort_values("year").reset_index(drop=True)
    abs_mat = d[PANEL_CHANNELS].abs().to_numpy()       # magnitude per channel
    denom = abs_mat.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    shares = abs_mat / denom                           # normalise each entry to 100%
    years = d["year"].to_numpy() + 1
    countries = d["entered_country"].to_numpy() + 1
    bottom = np.zeros(len(d))
    for j, ch in enumerate(PANEL_CHANNELS):            # stack channel shares per entry
        ax.bar(years, shares[:, j], bottom=bottom, width=0.8,
               color=color_for[ch], edgecolor="white", linewidth=0.4)
        bottom += shares[:, j]
    for i in range(len(d)):                            # label each bar with entered country
        ax.text(years[i], 1.02, f"c{int(countries[i])}", ha="center",
                va="bottom", fontsize=6, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SHAP share (Focal firm A)")
    ax.set_title("Market entry decision: |SHAP| channel composition", loc="center", pad=18)
    handles = [Patch(facecolor=color_for[c], label=c) for c in PANEL_CHANNELS]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=11, title="SHAP channel")


## Figure generation
# Create overall figure
def make_figure(seed, rng, title=None):
    print(f"\nseed {seed}")
    fig, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    draw_presence(axes[0], load_grid(seed)) 
    draw_ratio(axes[1], load_ratio(seed)) 
    draw_shap(axes[2], compute_decision_channels(seed, rng)) 

    axes[2].set_xlim(0.5, N_YEARS + 0.5) 
    axes[2].set_xticks(np.arange(1, N_YEARS + 1))
    axes[2].set_xticklabels([str(y) for y in np.arange(1, N_YEARS + 1)], fontsize=6)
    axes[2].set_xlabel("Year")

    fig.suptitle(title if title else f"seed {seed}", y=0.995, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"seed_{seed}_expansion_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved to {out}")

# Main command
def main():
    rng = np.random.default_rng(RANDOM_STATE) 
    for entry in SEEDS:
        seed, title = entry if isinstance(entry, (tuple, list)) else (entry, None)
        make_figure(seed, rng, title)

# Execute
if __name__ == "__main__":
    main()