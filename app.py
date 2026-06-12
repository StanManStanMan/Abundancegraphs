"""
Urban Score × Species Composition Analysis
Run with: streamlit run app.py
Requires: pip install streamlit pandas matplotlib scipy scikit-bio openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import braycurtis
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Optional: scikit-bio for PERMANOVA ─────────────────────────────────────
try:
    from skbio.stats.distance import permanova, DistanceMatrix
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False

# ── Colour helper — works on all matplotlib versions ───────────────────────
def get_colors(n):
    try:
        cmap = matplotlib.colormaps["tab20"]
    except AttributeError:
        cmap = matplotlib.cm.get_cmap("tab20")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]

# ── Safe linear regression — skips if all x values are identical ───────────
def safe_linregress(x, y):
    """Returns (slope, intercept, r, p) or None if regression is impossible."""
    if len(x) < 3 or x.nunique() < 2:
        return None
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope, intercept, r, p

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Urban Score × Community", layout="wide")
st.title("🌆 Urban Score × Species Composition")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH = "blasted_with_sites.xlsx"

@st.cache_data
def load_data(path):
    return pd.read_excel(path)

try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    uploaded = st.file_uploader("Upload blasted_with_sites.xlsx", type="xlsx")
    if uploaded is None:
        st.stop()
    df_raw = pd.read_excel(uploaded)

# ═══════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════════════════
SITE_COL  = "sites"
URBAN_COL = "urban score"
READ_COL  = "total supporting reads"
TAX_OPTIONS = {
    "Species":  "blast_species",
    "Genus":    "tax_genus",
    "Family":   "tax_family",
    "Order":    "tax_order",
    "Class":    "tax_class",
    "Phylum":   "tax_phylum",
    "Kingdom":  "tax_kingdom",
}

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONTROLS
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Settings")

tax_label = st.sidebar.selectbox(
    "Taxonomic level",
    options=list(TAX_OPTIONS.keys()),
    index=0,
)
tax_col = TAX_OPTIONS[tax_label]

sample_type = st.sidebar.radio(
    "Sample type",
    options=["A", "D", "Both"],
    index=2,
)

top_n = st.sidebar.slider(
    f"Top N {tax_label.lower()}s to display",
    min_value=3, max_value=20, value=8,
)

min_reads = st.sidebar.number_input(
    "Min reads per record (filter noise)",
    min_value=0, value=0, step=10,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Site grouping")

use_base = st.sidebar.radio(
    "Site labels",
    options=["Original (e.g. 6A1, 6A2, 6A3)", "Combined (e.g. 6A)"],
    index=0,
)
combine_sites = use_base == "Combined (e.g. 6A)"

if combine_sites:
    combine_method = st.sidebar.radio(
        "Combine replicates by",
        options=["Sum reads", "Average relative abundance"],
        index=0,
    )
else:
    combine_method = None

# ═══════════════════════════════════════════════════════════════════════════
# HELPER: strip trailing digits → base site label
# ═══════════════════════════════════════════════════════════════════════════
df_raw["site_base"] = df_raw[SITE_COL].str.replace(r"\d+$", "", regex=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def filter_sample_type(df, stype):
    if stype == "Both":
        return df[df[SITE_COL].str.contains("A|D")]
    return df[df[SITE_COL].str.contains(stype)]


def prepare_agg(df_in, tax_col, combine_sites, combine_method):
    df = df_in.copy()
    df = df.dropna(subset=[URBAN_COL, tax_col])
    df[tax_col] = df[tax_col].astype(str).str.strip()

    group_col = "site_base" if combine_sites else SITE_COL

    site_urban = (
        df.groupby(group_col)[URBAN_COL]
        .mean()
        .reset_index()
        .rename(columns={URBAN_COL: "urban_score", group_col: "site_id"})
    )

    if combine_sites and combine_method == "Sum reads":
        agg = (
            df.groupby([group_col, tax_col])[READ_COL]
            .sum()
            .reset_index()
            .rename(columns={READ_COL: "reads", group_col: "site_id"})
        )
        agg = agg.merge(site_urban, on="site_id")
        agg["rel_abund"] = agg.groupby("site_id")["reads"].transform(
            lambda x: x / x.sum()
        )

    elif combine_sites and combine_method == "Average relative abundance":
        df["rel_abund_orig"] = df.groupby(SITE_COL)[READ_COL].transform(
            lambda x: x / x.sum()
        )
        agg = (
            df.groupby([group_col, tax_col])
            .agg(reads=(READ_COL, "sum"), rel_abund=("rel_abund_orig", "mean"))
            .reset_index()
            .rename(columns={group_col: "site_id"})
        )
        agg = agg.merge(site_urban, on="site_id")

    else:
        agg = (
            df.groupby([SITE_COL, tax_col])[READ_COL]
            .sum()
            .reset_index()
            .rename(columns={READ_COL: "reads", SITE_COL: "site_id"})
        )
        agg = agg.merge(site_urban, on="site_id")
        agg["rel_abund"] = agg.groupby("site_id")["reads"].transform(
            lambda x: x / x.sum()
        )

    return agg


# ── Apply filters ──────────────────────────────────────────────────────────
df_filtered = filter_sample_type(df_raw, sample_type)
if min_reads > 0:
    df_filtered = df_filtered[df_filtered[READ_COL] >= min_reads]

agg = prepare_agg(df_filtered, tax_col, combine_sites, combine_method)

site_label_name = "Combined site" if combine_sites else "Site"
mode_str = f"combined {combine_method.lower()}" if combine_sites else "original sites"

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — MOST ABUNDANT TAXA OVER URBAN SCORE
# ═══════════════════════════════════════════════════════════════════════════
st.header(f"1 · Top {top_n} {tax_label}s vs. Urban Score")

top_taxa  = agg.groupby(tax_col)["reads"].sum().nlargest(top_n).index.tolist()
agg_top   = agg[agg[tax_col].isin(top_taxa)].copy()
colors    = get_colors(len(top_taxa))
color_map = dict(zip(top_taxa, colors))

fig1, ax1 = plt.subplots(figsize=(10, 5))
for taxon in top_taxa:
    sub = agg_top[agg_top[tax_col] == taxon].sort_values("urban_score")
    ax1.scatter(sub["urban_score"], sub["rel_abund"],
                color=color_map[taxon], alpha=0.5, s=40)
    result = safe_linregress(sub["urban_score"], sub["rel_abund"])
    if result:
        slope, intercept, r, p = result
        x_line = np.linspace(sub["urban_score"].min(), sub["urban_score"].max(), 100)
        ax1.plot(x_line, slope * x_line + intercept,
                 color=color_map[taxon], linewidth=1.8,
                 label=f"{taxon} (r={r:.2f}, p={p:.3f})")
    else:
        ax1.scatter([], [], color=color_map[taxon],
                    label=f"{taxon} (insufficient range)")

ax1.set_xlabel("Urban Score", fontsize=12)
ax1.set_ylabel("Relative Abundance", fontsize=12)
ax1.set_title(
    f"Top {top_n} {tax_label.lower()}s vs. urban score  [{sample_type} | {mode_str}]",
    fontsize=13,
)
ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.7)
ax1.set_xlim(0, 1)
fig1.tight_layout()
st.pyplot(fig1)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — SEPARATE A vs D PLOTS
# ═══════════════════════════════════════════════════════════════════════════
if sample_type == "Both":
    st.header(f"2 · A vs D Samples — Top {top_n} {tax_label}s")

    fig2, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, stype in zip(axes, ["A", "D"]):
        df_s = filter_sample_type(df_raw, stype)
        if min_reads > 0:
            df_s = df_s[df_s[READ_COL] >= min_reads]
        agg_s = prepare_agg(df_s, tax_col, combine_sites, combine_method)

        top_s       = agg_s.groupby(tax_col)["reads"].sum().nlargest(top_n).index.tolist()
        agg_s_top   = agg_s[agg_s[tax_col].isin(top_s)]
        colors_s    = get_colors(len(top_s))
        color_map_s = dict(zip(top_s, colors_s))

        for taxon in top_s:
            sub = agg_s_top[agg_s_top[tax_col] == taxon].sort_values("urban_score")
            ax.scatter(sub["urban_score"], sub["rel_abund"],
                       color=color_map_s[taxon], alpha=0.5, s=40)
            result = safe_linregress(sub["urban_score"], sub["rel_abund"])
            if result:
                slope, intercept, r, p = result
                x_line = np.linspace(sub["urban_score"].min(),
                                     sub["urban_score"].max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        color=color_map_s[taxon], linewidth=1.8,
                        label=f"{taxon} (r={r:.2f}, p={p:.3f})")
            else:
                ax.scatter([], [], color=color_map_s[taxon],
                           label=f"{taxon} (insufficient range)")

        ax.set_title(f"{stype} samples", fontsize=13)
        ax.set_xlabel("Urban Score", fontsize=11)
        ax.set_ylabel("Relative Abundance", fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=7, framealpha=0.7)

    fig2.suptitle(
        f"Top {top_n} {tax_label.lower()}s vs. urban score — A vs D  [{mode_str}]",
        fontsize=14, y=1.02,
    )
    fig2.tight_layout()
    st.pyplot(fig2)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════
st.header("3 · Significance of Urbanisation on Community Composition")

pivot = agg.pivot_table(
    index="site_id", columns=tax_col, values="rel_abund", fill_value=0
)
site_urban_df = agg[["site_id", "urban_score"]].drop_duplicates().set_index("site_id")
pivot = pivot.merge(site_urban_df, left_index=True, right_index=True)
pivot = pivot.dropna(subset=["urban_score"])

sites_ordered = pivot.index.tolist()
urban_vals    = pivot["urban_score"].values
comm_matrix   = pivot.drop(columns=["urban_score"]).values

n          = len(sites_ordered)
bc_matrix  = np.zeros((n, n))
urban_dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            bc_matrix[i, j]  = braycurtis(comm_matrix[i], comm_matrix[j])
            urban_dist[i, j] = abs(urban_vals[i] - urban_vals[j])

triu_idx      = np.triu_indices(n, k=1)
bc_flat       = bc_matrix[triu_idx]
urban_flat    = urban_dist[triu_idx]
rho, p_mantel = spearmanr(urban_flat, bc_flat)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mantel-like Test")
    st.markdown(
        f"""
        Spearman correlation between pairwise **urban score distance**
        and **Bray-Curtis dissimilarity**:

        | Metric | Value |
        |--------|-------|
        | ρ (Spearman) | **{rho:.4f}** |
        | p-value | **{p_mantel:.4f}** |
        | n (site pairs) | {len(bc_flat)} |

        {"✅ Significant (p < 0.05)" if p_mantel < 0.05 else "❌ Not significant (p ≥ 0.05)"}
        """
    )

with col2:
    st.subheader("PERMANOVA")
    if SKBIO_AVAILABLE:
        dm   = DistanceMatrix(bc_matrix, ids=sites_ordered)
        meta = pd.DataFrame({"urban_score": urban_vals}, index=sites_ordered)
        meta["urban_group"] = pd.cut(
            meta["urban_score"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
        )
        meta      = meta.dropna(subset=["urban_group"])
        valid_ids = meta.index.tolist()
        dm_sub    = dm.filter(valid_ids)
        result    = permanova(dm_sub, meta.loc[valid_ids], column="urban_group",
                              permutations=999)
        st.markdown(
            f"""
            PERMANOVA on Bray-Curtis dissimilarity,
            grouped by urban score tertile (Low / Medium / High):

            | Metric | Value |
            |--------|-------|
            | pseudo-F | **{result['test statistic']:.4f}** |
            | p-value | **{result['p-value']:.4f}** |
            | Permutations | 999 |

            {"✅ Significant (p < 0.05)" if result['p-value'] < 0.05 else "❌ Not significant (p ≥ 0.05)"}
            """
        )
    else:
        st.info(
            "Install **scikit-bio** for PERMANOVA:\n"
            "```\npip install scikit-bio\n```\n\n"
            "The Mantel-like Spearman test (left) is available without it."
        )

st.subheader("Bray-Curtis Dissimilarity vs. Urban Score Distance")
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.scatter(urban_flat, bc_flat, alpha=0.4, s=20, color="steelblue")
m, b, _, _, _ = stats.linregress(urban_flat, bc_flat)
x_r = np.linspace(0, urban_flat.max(), 200)
ax3.plot(x_r, m * x_r + b, color="tomato", linewidth=2,
         label=f"ρ = {rho:.3f}, p = {p_mantel:.4f}")
ax3.set_xlabel("|Urban score_i − Urban score_j|", fontsize=11)
ax3.set_ylabel("Bray-Curtis dissimilarity", fontsize=11)
ax3.set_title("Community dissimilarity vs. urban gradient", fontsize=12)
ax3.legend(fontsize=10)
fig3.tight_layout()
st.pyplot(fig3)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — STACKED BAR
# ═══════════════════════════════════════════════════════════════════════════
st.header("4 · Community Composition Along Urban Gradient")

pivot_sorted = pivot.sort_values("urban_score")
comm_sorted  = pivot_sorted.drop(columns=["urban_score"])
urban_sorted = pivot_sorted["urban_score"].values

top_bar  = agg.groupby(tax_col)["reads"].sum().nlargest(top_n).index.tolist()
comm_bar = comm_sorted[[c for c in comm_sorted.columns if c in top_bar]].copy()
comm_bar["Other"] = comm_sorted[
    [c for c in comm_sorted.columns if c not in top_bar]
].sum(axis=1)

bar_colors = get_colors(len(comm_bar.columns))
fig4, ax4  = plt.subplots(figsize=(12, 5))
bottom     = np.zeros(len(comm_bar))

for i, col in enumerate(comm_bar.columns):
    ax4.bar(range(len(comm_bar)), comm_bar[col].values,
            bottom=bottom, color=bar_colors[i], label=col, width=0.9)
    bottom += comm_bar[col].values

ax4.set_xticks(range(len(comm_bar)))
ax4.set_xticklabels(
    [f"{s}\n({u:.2f})" for s, u in zip(pivot_sorted.index, urban_sorted)],
    rotation=90, fontsize=7,
)
ax4.set_ylabel("Relative Abundance", fontsize=11)
ax4.set_xlabel(f"{site_label_name} (urban score)", fontsize=11)
ax4.set_title(
    f"Community composition sorted by urban score  [{sample_type} | {mode_str}]",
    fontsize=12,
)
ax4.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.7)
fig4.tight_layout()
st.pyplot(fig4)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — RAW DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════
with st.expander("📋 Show aggregated data table"):
    st.dataframe(
        agg.sort_values("urban_score").reset_index(drop=True),
        use_container_width=True,
    )
