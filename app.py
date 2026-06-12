"""
Combined: Urban Score x Species Composition + Nematode Community Explorer
Run with: streamlit run app.py
Requires: pip install streamlit pandas numpy matplotlib scipy scikit-learn scikit-bio openpyxl
"""

import re
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu, spearmanr
from scipy.spatial.distance import braycurtis, pdist, squareform
from sklearn.manifold import MDS
import streamlit as st

try:
    from skbio.stats.distance import permanova as skbio_permanova, DistanceMatrix
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Nematode Community Explorer", page_icon="🪱",
                   layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_colors(n):
    try:
        cmap = matplotlib.colormaps["tab20"]
    except AttributeError:
        cmap = matplotlib.cm.get_cmap("tab20")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]

def safe_linregress(x, y):
    if len(x) < 3 or x.nunique() < 2:
        return None
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope, intercept, r, p

def shannon_index(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))

def compute_relative_composition(df, taxon_col, reads_col):
    rel_dict = {}
    for (site, tr), sub in df.groupby(["_site", "_treatment"]):
        grp = sub.groupby(taxon_col)[reads_col].sum()
        total = grp.sum()
        rel_dict[(site, tr)] = grp / total * 100.0 if total > 0 else grp * 0.0
    return rel_dict

def make_color_map(taxa):
    c1 = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    c2 = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))
    c3 = plt.get_cmap("tab20c")(np.linspace(0, 1, 20))
    all_c = np.vstack([c1, c2, c3])
    return {t: all_c[i % len(all_c)] for i, t in enumerate(sorted(taxa))}

def bc_sim(va, vb):
    a, b = np.asarray(va, float), np.asarray(vb, float)
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    return (1 - braycurtis(a, b)) * 100.0

def sig_stars(p):
    if np.isnan(p): return "ns"
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"

def build_community_matrix(df, taxon_col, reads_col, combine_reps=False):
    records, labels = [], []
    if combine_reps:
        for (site, tr), sub in df.groupby(["_site", "_treatment"]):
            grp = sub.groupby(taxon_col)[reads_col].sum()
            total = grp.sum()
            rel = grp / total * 100.0 if total > 0 else grp * 0.0
            records.append(rel)
            labels.append(f"{site}{tr}")
    else:
        for (site, tr, rep), sub in df.groupby(["_site", "_treatment", "_rep"]):
            grp = sub.groupby(taxon_col)[reads_col].sum()
            total = grp.sum()
            rel = grp / total * 100.0 if total > 0 else grp * 0.0
            records.append(rel)
            labels.append(f"{site}{tr}{rep}")
    return pd.DataFrame(records).fillna(0), labels

def permanova_custom(dist_matrix, grouping, n_perm=999):
    groups = np.asarray(grouping)
    n = len(groups)
    k = len(np.unique(groups))
    def _f(d, g):
        ss_tot = np.sum(d**2) / n
        ss_w = 0.0
        for lbl in np.unique(g):
            idx = np.where(g == lbl)[0]
            ni = len(idx)
            if ni < 2: continue
            ss_w += np.sum(d[np.ix_(idx, idx)]**2) / ni
        dfa = k - 1
        dfw = n - k
        if dfw == 0 or ss_w == 0: return np.nan
        return ((ss_tot - ss_w) / dfa) / (ss_w / dfw)
    f_obs = _f(dist_matrix, groups)
    if np.isnan(f_obs): return f_obs, np.nan
    rng = np.random.default_rng(42)
    count = sum(1 for _ in range(n_perm)
                if _f(dist_matrix, rng.permutation(groups)) >= f_obs)
    return f_obs, (count + 1) / (n_perm + 1)

def run_mds(dist_matrix):
    mds = MDS(n_components=2, dissimilarity="precomputed",
              max_iter=500, n_init=10, random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(dist_matrix)
    return coords, mds.stress_

def confidence_ellipse(xs, ys, ax, n_std=2.0, **kw):
    from matplotlib.patches import Ellipse
    if len(xs) < 3: return
    cov = np.cov(xs, ys)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(xy=(np.mean(xs), np.mean(ys)),
                         width=w, height=h, angle=angle, **kw))

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

def parse_labels(df, regex, tr_a_chars, tr_d_chars):
    pattern = re.compile(regex)
    sites, treatments, reps = [], [], []
    for lbl in df["_label"].astype(str):
        m = pattern.match(lbl)
        if not m:
            raise ValueError(f"Label '{lbl}' does not match regex:\n{regex}")
        sites.append(int(m.group("site")))
        treatments.append(m.group("treatment"))
        reps.append(int(m.group("rep")))
    df = df.copy()
    df["_site"] = sites
    df["_treatment"] = treatments
    df["_rep"] = reps
    a_set = set(c.strip() for c in tr_a_chars.split(","))
    d_set = set(c.strip() for c in tr_d_chars.split(","))
    df["_treatment"] = df["_treatment"].map(
        lambda t: "A" if t in a_set else ("D" if t in d_set else t.upper()))
    return df

def first_match(candidates, pool, fallback="(none)"):
    return next((c for c in candidates if c in pool), fallback)

@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

@st.cache_data
def load_file(file_bytes, file_name):
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))
# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🌆 Urban Score x Species Composition", "🪱 Community Explorer"])

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — Urban Score x Species Composition                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.title("🌆 Urban Score x Species Composition")

    SITE_COL  = "sites"
    URBAN_COL = "urban score"
    READ_COL  = "total supporting reads"
    TAX_OPTIONS = {
        "Species": "blast_species", "Genus": "tax_genus", "Family": "tax_family",
        "Order": "tax_order", "Class": "tax_class", "Phylum": "tax_phylum", "Kingdom": "tax_kingdom",
    }

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        df_raw1 = load_excel("blasted_with_sites.xlsx")
    except FileNotFoundError:
        up1 = st.file_uploader("Upload blasted_with_sites.xlsx", type="xlsx", key="up1")
        if up1 is None:
            st.info("Upload your data file to get started.")
            st.stop()
        df_raw1 = pd.read_excel(up1)

    # Remove 0D0 controls
    df_raw1 = df_raw1[~df_raw1["sites"].str.upper().str.startswith("0D0")]

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.header("🌆 Urban Score Settings")

        tax_label = st.selectbox("Taxonomic level", list(TAX_OPTIONS.keys()),
                                 index=0, key="t1_tax")
        tax_col = TAX_OPTIONS[tax_label]

        sample_type = st.radio("Sample type", ["A", "D", "Both"], index=2, key="t1_stype")

        top_n = st.slider(f"Top N {tax_label.lower()}s", 3, 20, 8, key="t1_topn")

        min_reads = st.number_input("Min reads per record", 0, value=0, step=10, key="t1_minr")

        st.markdown("---")
        st.subheader("🔬 Taxonomic filter")
        filter_tax_label = st.selectbox("Filter by taxonomic level", list(TAX_OPTIONS.keys()),
                                        index=list(TAX_OPTIONS.keys()).index("Kingdom"),
                                        key="t1_ftax")
        filter_tax_col = TAX_OPTIONS[filter_tax_label]
        available_taxa = sorted(df_raw1[filter_tax_col].dropna().astype(str).str.strip().unique().tolist())
        selected_taxa = st.multiselect(f"Include only these {filter_tax_label.lower()}s",
                                       options=available_taxa, default=available_taxa, key="t1_stax")

        st.markdown("---")
        st.subheader("📍 Site grouping")
        use_base = st.radio("Site labels",
                            ["Original (e.g. 6A1, 6A2, 6A3)", "Combined (e.g. 6A)"],
                            index=0, key="t1_usebase")
        combine_sites = use_base == "Combined (e.g. 6A)"
        combine_method = None
        if combine_sites:
            combine_method = st.radio("Combine replicates by",
                                      ["Sum reads", "Average relative abundance"],
                                      index=0, key="t1_combm")

        st.markdown("---")
        st.subheader("🚫 Control sites")
        df_raw1["site_base"] = df_raw1[SITE_COL].str.replace(r"\d+$", "", regex=True)
        all_site_bases = sorted(df_raw1["site_base"].unique().tolist())
        exclude_controls = st.multiselect(
            "Exclude these sites", options=all_site_bases,
            default=[s for s in all_site_bases if "0D0" in s or "0A0" in s],
            key="t1_excl")

    # ── Apply filters ──────────────────────────────────────────────────────
    if not selected_taxa:
        st.warning("No taxa selected — please select at least one group.")
        st.stop()
    df_raw1 = df_raw1[df_raw1[filter_tax_col].astype(str).str.strip().isin(selected_taxa)]
    if exclude_controls:
        df_raw1 = df_raw1[~df_raw1["site_base"].isin(exclude_controls)]

    def filter_stype(df, stype):
        if stype == "Both": return df[df[SITE_COL].str.contains("A|D")]
        return df[df[SITE_COL].str.contains(stype)]

    def prepare_agg(df_in, tc, cs, cm):
        df = df_in.copy().dropna(subset=[URBAN_COL, tc])
        df[tc] = df[tc].astype(str).str.strip()
        gc = "site_base" if cs else SITE_COL
        su = df.groupby(gc)[URBAN_COL].mean().reset_index().rename(
            columns={URBAN_COL: "urban_score", gc: "site_id"})
        if cs and cm == "Sum reads":
            agg = df.groupby([gc, tc])[READ_COL].sum().reset_index().rename(
                columns={READ_COL: "reads", gc: "site_id"})
            agg = agg.merge(su, on="site_id")
            agg["rel_abund"] = agg.groupby("site_id")["reads"].transform(lambda x: x / x.sum())
        elif cs and cm == "Average relative abundance":
            df["rel_abund_orig"] = df.groupby(SITE_COL)[READ_COL].transform(lambda x: x / x.sum())
            agg = df.groupby([gc, tc]).agg(reads=(READ_COL, "sum"),
                                           rel_abund=("rel_abund_orig", "mean")).reset_index()
            agg = agg.rename(columns={gc: "site_id"}).merge(su, on="site_id")
        else:
            agg = df.groupby([SITE_COL, tc])[READ_COL].sum().reset_index().rename(
                columns={READ_COL: "reads", SITE_COL: "site_id"})
            agg = agg.merge(su, on="site_id")
            agg["rel_abund"] = agg.groupby("site_id")["reads"].transform(lambda x: x / x.sum())
        return agg

    df_f1 = filter_stype(df_raw1, sample_type)
    if min_reads > 0:
        df_f1 = df_f1[df_f1[READ_COL] >= min_reads]
    agg = prepare_agg(df_f1, tax_col, combine_sites, combine_method)
    site_label_name = "Combined site" if combine_sites else "Site"
    mode_str = f"combined {combine_method.lower()}" if combine_sites else "original sites"

    with st.expander("🔎 Active filters", expanded=False):
        st.markdown(
            f"- **Level:** {tax_label} | **Filter:** {filter_tax_label} in {selected_taxa}\n"
            f"- **Sample type:** {sample_type} | **Mode:** {mode_str} | **Min reads:** {min_reads}\n"
            f"- **Excluded:** {exclude_controls if exclude_controls else 'None'}")

    # ── Section 1: Top N taxa vs urban score ──────────────────────────────
    st.header(f"1 · Top {top_n} {tax_label}s vs. Urban Score")
    top_taxa = agg.groupby(tax_col)["reads"].sum().nlargest(top_n).index.tolist()
    agg_top  = agg[agg[tax_col].isin(top_taxa)].copy()
    cmap1    = dict(zip(top_taxa, get_colors(len(top_taxa))))
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for taxon in top_taxa:
        sub = agg_top[agg_top[tax_col] == taxon].sort_values("urban_score")
        ax1.scatter(sub["urban_score"], sub["rel_abund"], color=cmap1[taxon], alpha=0.5, s=40)
        res = safe_linregress(sub["urban_score"], sub["rel_abund"])
        if res:
            sl, ic, r, p = res
            xl = np.linspace(sub["urban_score"].min(), sub["urban_score"].max(), 100)
            ax1.plot(xl, sl*xl+ic, color=cmap1[taxon], linewidth=1.8,
                     label=f"{taxon} (r={r:.2f}, p={p:.3f})")
        else:
            ax1.scatter([], [], color=cmap1[taxon], label=f"{taxon} (insufficient range)")
    ax1.set_xlabel("Urban Score", fontsize=12)
    ax1.set_ylabel("Relative Abundance", fontsize=12)
    ax1.set_title(f"Top {top_n} {tax_label.lower()}s vs. urban score  [{sample_type} | {mode_str}]", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.7)
    ax1.set_xlim(0, 1)
    fig1.tight_layout()
    st.pyplot(fig1)

    # ── Section 2: A vs D split ────────────────────────────────────────────
    if sample_type == "Both":
        st.header(f"2 · A vs D Samples — Top {top_n} {tax_label}s")
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for ax2, stype in zip(axes2, ["A", "D"]):
            df_s = filter_stype(df_raw1, stype)
            if min_reads > 0: df_s = df_s[df_s[READ_COL] >= min_reads]
            agg_s = prepare_agg(df_s, tax_col, combine_sites, combine_method)
            top_s = agg_s.groupby(tax_col)["reads"].sum().nlargest(top_n).index.tolist()
            agg_st = agg_s[agg_s[tax_col].isin(top_s)]
            cs2 = dict(zip(top_s, get_colors(len(top_s))))
            for taxon in top_s:
                sub = agg_st[agg_st[tax_col] == taxon].sort_values("urban_score")
                ax2.scatter(sub["urban_score"], sub["rel_abund"], color=cs2[taxon], alpha=0.5, s=40)
                res = safe_linregress(sub["urban_score"], sub["rel_abund"])
                if res:
                    sl, ic, r, p = res
                    xl = np.linspace(sub["urban_score"].min(), sub["urban_score"].max(), 100)
                    ax2.plot(xl, sl*xl+ic, color=cs2[taxon], linewidth=1.8,
                             label=f"{taxon} (r={r:.2f}, p={p:.3f})")
                else:
                    ax2.scatter([], [], color=cs2[taxon], label=f"{taxon} (insufficient range)")
            ax2.set_title(f"{stype} samples", fontsize=13)
            ax2.set_xlabel("Urban Score", fontsize=11)
            ax2.set_ylabel("Relative Abundance", fontsize=11)
            ax2.set_xlim(0, 1)
            ax2.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, framealpha=0.7)
        fig2.suptitle(f"Top {top_n} {tax_label.lower()}s vs. urban score — A vs D  [{mode_str}]",
                      fontsize=14, y=1.02)
        fig2.tight_layout()
        st.pyplot(fig2)

    # ── Section 3: Significance ────────────────────────────────────────────
    st.header("3 · Significance of Urbanisation on Community Composition")
    pivot = agg.pivot_table(index="site_id", columns=tax_col, values="rel_abund", fill_value=0)
    su_df = agg[["site_id", "urban_score"]].drop_duplicates().set_index("site_id")
    pivot = pivot.merge(su_df, left_index=True, right_index=True).dropna(subset=["urban_score"])
    so = pivot.index.tolist()
    uv = pivot["urban_score"].values
    cm_arr = pivot.drop(columns=["urban_score"]).values
    n = len(so)
    bc_mat = np.zeros((n, n))
    ud_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                bc_mat[i, j] = braycurtis(cm_arr[i], cm_arr[j])
                ud_mat[i, j] = abs(uv[i] - uv[j])
    ti = np.triu_indices(n, k=1)
    bc_flat = bc_mat[ti]
    uf_flat = ud_mat[ti]
    rho, p_mantel = spearmanr(uf_flat, bc_flat)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Mantel-like Test")
        st.markdown(f"""
Spearman correlation between pairwise **urban score distance** and **Bray-Curtis dissimilarity**:

| Metric | Value |
|--------|-------|
| rho (Spearman) | **{rho:.4f}** |
| p-value | **{p_mantel:.4f}** |
| n (site pairs) | {len(bc_flat)} |

{"✅ Significant (p < 0.05)" if p_mantel < 0.05 else "❌ Not significant (p >= 0.05)"}
""")
    with col_b:
        st.subheader("PERMANOVA")
        if SKBIO_AVAILABLE:
            dm = DistanceMatrix(bc_mat, ids=so)
            meta = pd.DataFrame({"urban_score": uv}, index=so)
            meta["urban_group"] = pd.cut(meta["urban_score"],
                                         bins=[0, 0.33, 0.66, 1.0],
                                         labels=["Low", "Medium", "High"])
            meta = meta.dropna(subset=["urban_group"])
            vid = meta.index.tolist()
            res_p = skbio_permanova(dm.filter(vid), meta.loc[vid],
                                    column="urban_group", permutations=999)
            st.markdown(f"""
PERMANOVA on Bray-Curtis, grouped by urban score tertile:

| Metric | Value |
|--------|-------|
| pseudo-F | **{res_p['test statistic']:.4f}** |
| p-value | **{res_p['p-value']:.4f}** |
| Permutations | 999 |

{"✅ Significant (p < 0.05)" if res_p['p-value'] < 0.05 else "❌ Not significant (p >= 0.05)"}
""")
        else:
            st.info("Install **scikit-bio** for PERMANOVA: `pip install scikit-bio`")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.scatter(uf_flat, bc_flat, alpha=0.4, s=20, color="steelblue")
    m3, b3, _, _, _ = stats.linregress(uf_flat, bc_flat)
    xr = np.linspace(0, uf_flat.max(), 200)
    ax3.plot(xr, m3*xr+b3, color="tomato", linewidth=2, label=f"rho={rho:.3f}, p={p_mantel:.4f}")
    ax3.set_xlabel("|Urban score_i - Urban score_j|", fontsize=11)
    ax3.set_ylabel("Bray-Curtis dissimilarity", fontsize=11)
    ax3.set_title("Community dissimilarity vs. urban gradient", fontsize=12)
    ax3.legend(fontsize=10)
    fig3.tight_layout()
    st.pyplot(fig3)

    # ── Section 4: Stacked bar ─────────────────────────────────────────────
    st.header("4 · Community Composition Along Urban Gradient")
    ps = pivot.sort_values("urban_score")
    cs_arr = ps.drop(columns=["urban_score"])
    us_arr = ps["urban_score"].values
    tb = agg.groupby(tax_col)["reads"].sum().nlargest(top_n).index.tolist()
    cb = cs_arr[[c for c in cs_arr.columns if c in tb]].copy()
    cb["Other"] = cs_arr[[c for c in cs_arr.columns if c not in tb]].sum(axis=1)
    bc4 = get_colors(len(cb.columns))
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    bot4 = np.zeros(len(cb))
    for i, col in enumerate(cb.columns):
        ax4.bar(range(len(cb)), cb[col].values, bottom=bot4, color=bc4[i], label=col, width=0.9)
        bot4 += cb[col].values
    ax4.set_xticks(range(len(cb)))
    ax4.set_xticklabels([f"{s}\n({u:.2f})" for s, u in zip(ps.index, us_arr)],
                        rotation=90, fontsize=7)
    ax4.set_ylabel("Relative Abundance", fontsize=11)
    ax4.set_xlabel(f"{site_label_name} (urban score)", fontsize=11)
    ax4.set_title(f"Community composition sorted by urban score  [{sample_type} | {mode_str}]", fontsize=12)
    ax4.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.7)
    fig4.tight_layout()
    st.pyplot(fig4)

    with st.expander("📋 Show aggregated data table"):
        st.dataframe(agg.sort_values("urban_score").reset_index(drop=True), use_container_width=True)
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — Community Explorer                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.title("🪱 Nematode Community Explorer")

    with st.sidebar:
        st.markdown("---")
        st.header("🪱 Community Explorer Settings")
        st.subheader("① Upload data")
        uploaded2 = st.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"], key="up2")

    if uploaded2 is None:
        st.info("👈 Upload your Excel or CSV file in the sidebar to get started.")
        st.stop()

    df_raw2 = load_file(uploaded2.read(), uploaded2.name)

    with st.sidebar:
        st.success(f"Loaded: {len(df_raw2):,} rows x {len(df_raw2.columns)} cols")
        all_cols2  = list(df_raw2.columns)
        none_cols2 = ["(none)"] + all_cols2
        num_cols2  = df_raw2.select_dtypes(include=[np.number]).columns.tolist()
        obj_cols2  = df_raw2.select_dtypes(include=["object"]).columns.tolist()

        st.subheader("② Column mapping")
        reads_col2  = st.selectbox("Read counts *", none_cols2, key="t2_rc",
            index=none_cols2.index(first_match(["total supporting reads","reads","count"], all_cols2)))
        label_col2  = st.selectbox("Sample label *", none_cols2, key="t2_lc",
            index=none_cols2.index(first_match(["sites","label","sample","sample_id"], all_cols2)))
        taxon_col2  = st.selectbox("Species / taxon *", none_cols2, key="t2_tc",
            index=none_cols2.index(first_match(["blast_species","species","taxon"], all_cols2)))
        phylum_col2 = st.selectbox("Phylum column", none_cols2, key="t2_pc",
            index=none_cols2.index(first_match(["tax_phylum","phylum"], all_cols2)))

        st.subheader("③ Label parsing")
        regex_val2  = st.text_input("Regex (named groups: site, treatment, rep)",
            value=r"^(?P<site>[0-9]+)(?P<treatment>[aAdD])(?P<rep>[0-9]+)$", key="t2_re")
        tr_a_chars2 = st.text_input("Treatment A chars (comma-sep)", value="a,A", key="t2_ta")
        tr_d_chars2 = st.text_input("Treatment D chars (comma-sep)", value="d,D", key="t2_td")

        st.subheader("④ Filters")
        if phylum_col2 != "(none)" and phylum_col2 in df_raw2.columns:
            avail_phyla2 = sorted(df_raw2[phylum_col2].dropna().astype(str).unique().tolist())
            sel_phyla2   = st.multiselect("Include phyla", avail_phyla2,
                                          default=avail_phyla2, key="t2_ph")
        else:
            sel_phyla2 = None
            st.info("Set a Phylum column in ② to enable phylum filtering.")
        sites_input2 = st.text_input("Sites to include (comma-sep, empty=all)", value="", key="t2_si")

        st.subheader("⑤ Plot type")
        plot_type2 = st.radio("Choose plot", [
            "Stacked bar  (A vs D)",
            "Shannon diversity",
            "Shannon vs environment",
            "NMDS / PCoA ordination",
            "Replicate similarity table",
        ], key="t2_pt")

    # ── Build working dataframe ────────────────────────────────────────────
    @st.cache_data
    def build_df2(fb, fn, rc, lc, tc, pc, sp, si, rv, ta, td):
        df = load_file(fb, fn)
        if rc == "(none)" or lc == "(none)" or tc == "(none)":
            return None, "Set Read counts, Sample label, and Species/taxon columns."
        df = df[pd.to_numeric(df[rc], errors="coerce").fillna(0) > 0].copy()
        if pc != "(none)" and pc in df.columns and sp is not None and len(sp) > 0:
            df = df[df[pc].astype(str).isin(sp)]
        df["_label"] = df[lc].astype(str)
        try:
            df = parse_labels(df, rv, ta, td)
        except Exception as e:
            return None, str(e)
        if si.strip():
            try:
                sites = [int(s.strip()) for s in si.replace(";", ",").split(",") if s.strip()]
                df = df[df["_site"].isin(sites)]
            except:
                return None, "Sites filter: enter comma-separated integers."
        if df.empty:
            return None, "All rows removed by filters."
        return df, None

    uploaded2.seek(0)
    raw2 = uploaded2.read()
    df2, err2 = build_df2(raw2, uploaded2.name, reads_col2, label_col2, taxon_col2, phylum_col2,
                          tuple(sel_phyla2) if sel_phyla2 is not None else None,
                          sites_input2, regex_val2, tr_a_chars2, tr_d_chars2)
    if err2:
        st.error(err2)
        st.stop()

    pb = f" · Phyla: **{', '.join(sel_phyla2)}**" if sel_phyla2 else ""
    st.caption(f"**{uploaded2.name}** — {len(df2):,} rows · {df2['_site'].nunique()} sites · "
               f"{df2['_treatment'].nunique()} treatments · {df2['_rep'].nunique()} replicates{pb}")

    tc2_cands = [c for c in [phylum_col2, taxon_col2] + obj_cols2
                 if c != "(none)" and c in df2.columns]
    tc2_cands = list(dict.fromkeys(tc2_cands))

    # ── Stacked bar ────────────────────────────────────────────────────────
    if plot_type2.startswith("Stacked"):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        comp_col = c1.selectbox("Composition level", tc2_cands,
            index=tc2_cands.index(taxon_col2) if taxon_col2 in tc2_cands else 0, key="t2_cc")
        min_pct  = c2.number_input("Min % to show", 0.0, 50.0, 1.0, 0.5, key="t2_mp")
        max_leg  = c3.number_input("Max taxa in legend", 1, 60, 20, 1, key="t2_ml")
        show_pct = c4.checkbox("% labels in bars", value=True, key="t2_sp")
        show_bc  = st.checkbox("Show Bray-Curtis similarity below bars", value=True, key="t2_sb")

        if comp_col not in df2.columns:
            st.error(f"Column '{comp_col}' not found.")
            st.stop()

        df_s2 = df2.dropna(subset=[comp_col])
        rd2   = compute_relative_composition(df_s2, comp_col, reads_col2)
        sites2 = sorted({s for (s, _) in rd2})
        at2    = sorted(set().union(*[r.index for r in rd2.values()]))
        cm2    = make_color_map(at2)
        ns2    = len(sites2)
        fw2    = max(8, ns2 * 1.4 + 4)

        if show_bc:
            fig_b, (ax_b, axb2) = plt.subplots(2, 1, figsize=(fw2, 7),
                gridspec_kw={"height_ratios": [6, 1], "hspace": 0.08})
        else:
            fig_b, ax_b = plt.subplots(figsize=(fw2, 6))
            axb2 = None

        bw2 = 0.35
        xs2 = np.arange(ns2)
        bots2 = {(s, tr): 0.0 for s in sites2 for tr in ["A", "D"]}
        for tax in at2:
            for i, s in enumerate(sites2):
                for tr, off in [("A", -bw2/2), ("D", bw2/2)]:
                    rel = rd2.get((s, tr))
                    h = float(rel.get(tax, 0.0)) if rel is not None else 0.0
                    if h < min_pct: continue
                    b = bots2[(s, tr)]
                    ax_b.bar(xs2[i]+off, h, bw2, bottom=b,
                             color=cm2[tax], edgecolor="white", linewidth=0.3)
                    if show_pct and h >= 3.0:
                        ax_b.text(xs2[i]+off, b+h/2, f"{h:.0f}%", ha="center", va="center",
                                  fontsize=6.5, color="white", fontweight="bold")
                    bots2[(s, tr)] = b + h
        for i, s in enumerate(sites2):
            ax_b.text(xs2[i]-bw2/2, 101.5, "A", ha="center", va="bottom",
                      fontsize=8, color="#2c7bb6", fontweight="bold")
            ax_b.text(xs2[i]+bw2/2, 101.5, "D", ha="center", va="bottom",
                      fontsize=8, color="#d7191c", fontweight="bold")
        ax_b.set_xticks(xs2)
        ax_b.set_xticklabels([f"Site {s}" for s in sites2], fontsize=10)
        ax_b.set_ylabel("Relative abundance (%)", fontsize=11)
        ax_b.set_ylim(0, 107)
        ax_b.set_title(f"Community composition (A vs D) — {comp_col}", fontsize=13, fontweight="bold")
        ax_b.grid(axis="y", linestyle="--", alpha=0.3)
        ax_b.spines[["top", "right"]].set_visible(False)
        ta2 = {t: sum(float(r.get(t, 0)) for r in rd2.values()) for t in at2}
        tt2 = sorted(at2, key=lambda t: ta2[t], reverse=True)[:int(max_leg)]
        h2  = [mpatches.Patch(facecolor=cm2[t], edgecolor="white", linewidth=0.5, label=t) for t in tt2]
        ax_b.legend(handles=h2, title=f"Top {len(tt2)} taxa", title_fontsize=8, fontsize=7,
                    bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True, edgecolor="#ccc")

        if show_bc and axb2 is not None:
            axb2.set_xlim(ax_b.get_xlim()); axb2.set_ylim(0, 1); axb2.axis("off")
            for i, s in enumerate(sites2):
                ra = rd2.get((s, "A")); rd_d = rd2.get((s, "D"))
                if ra is None or rd_d is None: continue
                asp = sorted(set(ra.index) | set(rd_d.index))
                va2 = np.array([float(ra.get(sp, 0)) for sp in asp])
                vd2 = np.array([float(rd_d.get(sp, 0)) for sp in asp])
                sim2 = bc_sim(va2, vd2)
                rpa = df_s2[(df_s2["_site"]==s) & (df_s2["_treatment"]=="A")]
                rpd = df_s2[(df_s2["_site"]==s) & (df_s2["_treatment"]=="D")]
                def _hr(sub):
                    out = []
                    for _, rs in sub.groupby("_rep"):
                        grp = rs.groupby(comp_col)[reads_col2].sum()
                        out.append(shannon_index(grp.values))
                    return out
                ha2 = _hr(rpa); hd2 = _hr(rpd); pv2 = np.nan
                if len(ha2) >= 2 and len(hd2) >= 2:
                    try: _, pv2 = mannwhitneyu(ha2, hd2, alternative="two-sided")
                    except: pass
                sg2 = sig_stars(pv2)
                bs2 = f"BC: {sim2:.1f}%" if not np.isnan(sim2) else "BC: N/A"
                ps2 = f"  p={pv2:.3f}" if not np.isnan(pv2) else ""
                cc2 = ("#27ae60" if (not np.isnan(sim2) and sim2 >= 70) else
                       "#f39c12" if (not np.isnan(sim2) and sim2 >= 40) else "#c0392b")
                axb2.add_patch(mpatches.FancyBboxPatch(
                    (xs2[i]-0.42, 0.05), 0.84, 0.90, boxstyle="round,pad=0.02",
                    facecolor=cc2, edgecolor="white", alpha=0.85, linewidth=1,
                    transform=axb2.transData))
                axb2.text(xs2[i], 0.50, f"{bs2}  {sg2}{ps2}", ha="center", va="center",
                          fontsize=7.5, color="white", fontweight="bold", transform=axb2.transData)
            bl2 = [mpatches.Patch(color="#27ae60", label="BC >= 70%"),
                   mpatches.Patch(color="#f39c12", label="BC 40-70%"),
                   mpatches.Patch(color="#c0392b", label="BC < 40%")]
            axb2.legend(handles=bl2, loc="lower right", fontsize=7,
                        frameon=True, edgecolor="#ccc", bbox_to_anchor=(1.0, -0.1))
            axb2.set_title("A vs D  Bray-Curtis similarity  |  * p<0.05  ** p<0.01  *** p<0.001",
                           fontsize=8, color="#444", loc="left", pad=3)
        fig_b.tight_layout(rect=[0, 0, 0.82, 1])
        st.pyplot(fig_b, use_container_width=False)
        st.download_button("Download PNG", fig_to_bytes(fig_b),
                           file_name="stacked_bar.png", mime="image/png")

    # ── Shannon diversity ──────────────────────────────────────────────────
    elif plot_type2.startswith("Shannon div"):
        c1, c2 = st.columns([2, 2])
        sh2  = c1.selectbox("Diversity level", tc2_cands,
            index=tc2_cands.index(taxon_col2) if taxon_col2 in tc2_cands else 0, key="t2_sh")
        grp2 = c2.radio("Group by", [
            "site x treatment x replicate", "site x treatment",
            "treatment only", "site only"], key="t2_grp")
        recs2 = []
        if grp2 == "site x treatment x replicate":
            for (s, tr, rep), sub in df2.groupby(["_site", "_treatment", "_rep"]):
                grp = sub.groupby(sh2)[reads_col2].sum()
                recs2.append({"label": f"{s}{tr}{rep}", "treatment": tr,
                              "shannon": shannon_index(grp.values)})
        elif grp2 == "site x treatment":
            for (s, tr), sub in df2.groupby(["_site", "_treatment"]):
                grp = sub.groupby(sh2)[reads_col2].sum()
                recs2.append({"label": f"{s}{tr}", "treatment": tr,
                              "shannon": shannon_index(grp.values)})
        elif grp2 == "treatment only":
            for tr, sub in df2.groupby("_treatment"):
                grp = sub.groupby(sh2)[reads_col2].sum()
                recs2.append({"label": f"Tr. {tr}", "treatment": tr,
                              "shannon": shannon_index(grp.values)})
        else:
            for s, sub in df2.groupby("_site"):
                grp = sub.groupby(sh2)[reads_col2].sum()
                recs2.append({"label": f"Site {s}", "treatment": "?",
                              "shannon": shannon_index(grp.values)})
        out2 = pd.DataFrame(recs2).sort_values("label")
        tc2c = {"A": "#2c7bb6", "D": "#d7191c", "?": "#888888"}
        x2 = np.arange(len(out2))
        fig_sh, ax_sh = plt.subplots(figsize=(max(7, len(out2)*0.6+2), 5))
        bars2 = ax_sh.bar(x2, out2["shannon"],
                          color=[tc2c.get(t, "#888") for t in out2["treatment"]],
                          edgecolor="white", linewidth=0.5, width=0.7, zorder=2)
        for bar, val in zip(bars2, out2["shannon"]):
            if not np.isnan(val):
                ax_sh.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax_sh.set_xticks(x2)
        ax_sh.set_xticklabels(out2["label"], rotation=45, ha="right", fontsize=9)
        ax_sh.set_ylabel("Shannon index (H)", fontsize=11)
        ax_sh.set_title(f"Shannon diversity — {sh2}  [{grp2}]", fontsize=12, fontweight="bold")
        ax_sh.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
        ax_sh.spines[["top", "right"]].set_visible(False)
        ax_sh.legend(handles=[mpatches.Patch(color="#2c7bb6", label="Treatment A"),
                               mpatches.Patch(color="#d7191c", label="Treatment D")],
                     fontsize=9, frameon=True, edgecolor="#ccc")
        fig_sh.tight_layout()
        st.pyplot(fig_sh, use_container_width=False)
        st.download_button("Download PNG", fig_to_bytes(fig_sh),
                           file_name="shannon.png", mime="image/png")

    # ── Shannon vs environment ─────────────────────────────────────────────
    elif plot_type2.startswith("Shannon vs"):
        c1, _ = st.columns([2, 2])
        sh3 = c1.selectbox("Diversity level", tc2_cands,
            index=tc2_cands.index(taxon_col2) if taxon_col2 in tc2_cands else 0, key="t2_sh3")
        ec2 = ["(none)"] + num_cols2
        c3, c4, c5, c6 = st.columns(4)
        e1 = c3.selectbox("X-axis 1", ec2, index=1 if len(ec2) > 1 else 0, key="t2_e1")
        e2 = c4.selectbox("X-axis 2", ec2, index=0, key="t2_e2")
        e3 = c5.selectbox("X-axis 3", ec2, index=0, key="t2_e3")
        e4 = c6.selectbox("X-axis 4", ec2, index=0, key="t2_e4")
        ecols = [e for e in [e1, e2, e3, e4] if e != "(none)"]
        if not ecols:
            st.warning("Select at least one X-axis variable.")
            st.stop()
        recs3 = []
        for (s, tr), sub in df2.groupby(["_site", "_treatment"]):
            grp = sub.groupby(sh3)[reads_col2].sum()
            rec = {"site": s, "treatment": tr, "shannon": shannon_index(grp.values)}
            for col in ecols:
                rec[col] = pd.to_numeric(sub[col], errors="coerce").mean() if col in sub.columns else np.nan
            recs3.append(rec)
        out3 = pd.DataFrame(recs3)
        ne = len(ecols)
        fig_ev, axs_ev = plt.subplots(1, ne, figsize=(5*ne, 5), squeeze=False)
        axs_ev = axs_ev[0]
        sty3 = {"A": {"color": "#2c7bb6", "marker": "o"}, "D": {"color": "#d7191c", "marker": "s"}}
        for ax_e, env in zip(axs_ev, ecols):
            for tr, st3 in sty3.items():
                sub = out3[out3["treatment"] == tr]
                xe = pd.to_numeric(sub[env], errors="coerce").values
                ye = sub["shannon"].values
                if xe.size == 0: continue
                ax_e.scatter(xe, ye, color=st3["color"], marker=st3["marker"], s=120,
                             edgecolors="white", lw=0.8, label=f"Treatment {tr}", zorder=3)
                for _, row in sub.iterrows():
                    ax_e.annotate(str(int(row["site"])), xy=(row[env], row["shannon"]),
                                  xytext=(6, 4), textcoords="offset points",
                                  fontsize=9, fontweight="bold", color=st3["color"])
                mask = ~np.isnan(xe) & ~np.isnan(ye)
                if mask.sum() >= 3 and np.nanstd(xe[mask]) > 0:
                    xm, ym = xe[mask], ye[mask]
                    me, be = np.polyfit(xm, ym, 1)
                    xl = np.linspace(xm.min(), xm.max(), 100)
                    ax_e.plot(xl, me*xl+be, color=st3["color"], ls="--", lw=1.5, alpha=0.7)
                    re_v, pe = pearsonr(xm, ym)
                    ps3 = f"p={pe:.3f}" if pe >= 0.001 else "p<0.001"
                    off = 0.22 if tr == "A" else 0.05
                    ax_e.text(0.98, off, f"Tr.{tr}  r={re_v:.2f}, {ps3}",
                              transform=ax_e.transAxes, ha="right", va="bottom", fontsize=9,
                              color=st3["color"],
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=st3["color"], lw=1))
            ax_e.set_xlabel(env, fontsize=10)
            ax_e.set_ylabel("Shannon H" if env == ecols[0] else "", fontsize=10)
            ax_e.set_title(env, fontsize=11, fontweight="bold")
            ax_e.grid(True, ls="--", alpha=0.4)
            ax_e.spines[["top", "right"]].set_visible(False)
            if env == ecols[0]:
                ax_e.legend(fontsize=9, frameon=True, edgecolor="#ccc")
        fig_ev.suptitle(f"Shannon vs environment — {sh3}", fontsize=13, fontweight="bold")
        fig_ev.tight_layout()
        st.pyplot(fig_ev, use_container_width=False)
        st.download_button("Download PNG", fig_to_bytes(fig_ev),
                           file_name="shannon_env.png", mime="image/png")

    # ── NMDS ──────────────────────────────────────────────────────────────
    elif plot_type2.startswith("NMDS"):
        c1, c2, c3, c4 = st.columns(4)
        sh4   = c1.selectbox("Distance level", tc2_cands,
            index=tc2_cands.index(taxon_col2) if taxon_col2 in tc2_cands else 0, key="t2_sh4")
        cb4   = c2.radio("Colour by", ["treatment", "site", "rep"], key="t2_cb4")
        comb4 = c3.checkbox("Combine replicates", value=False, key="t2_comb4")
        ell4  = c3.checkbox("95% ellipses", value=True, key="t2_ell4")
        np4   = c4.number_input("PERMANOVA permutations", 99, 9999, 999, 100, key="t2_np4")

        mat4, labs4 = build_community_matrix(df2, sh4, reads_col2, combine_reps=comb4)
        if len(mat4) < 3:
            st.error("Need at least 3 samples.")
            st.stop()

        pat4 = (re.compile(r"^(?P<site>[0-9]+)(?P<treatment>[A-Z])$") if comb4
                else re.compile(regex_val2))
        as4 = set(c.strip() for c in tr_a_chars2.split(","))
        ds4 = set(c.strip() for c in tr_d_chars2.split(","))
        meta4 = []
        for lbl in labs4:
            m4 = pat4.match(lbl)
            if m4:
                tr4 = m4.group("treatment")
                tr4 = "A" if tr4 in as4 else ("D" if tr4 in ds4 else tr4.upper())
                r4  = int(m4.group("rep")) if "rep" in pat4.groupindex else 0
                meta4.append({"label": lbl, "site": int(m4.group("site")),
                              "treatment": tr4, "rep": r4})
            else:
                meta4.append({"label": lbl, "site": 0, "treatment": "?", "rep": 0})
        mdf4 = pd.DataFrame(meta4)
        dm4  = squareform(pdist(mat4.values, metric="braycurtis"))
        f4, p4 = permanova_custom(dm4, mdf4["treatment"].values, n_perm=int(np4))
        co4, st4 = run_mds(dm4)

        if cb4 == "treatment":
            pal4 = {"A": "#2c7bb6", "D": "#d7191c"}
            gv4  = mdf4["treatment"].tolist()
        elif cb4 == "site":
            su4  = sorted(mdf4["site"].unique())
            cm4  = plt.get_cmap("tab10")
            pal4 = {s: cm4(i/max(len(su4)-1, 1)) for i, s in enumerate(su4)}
            gv4  = mdf4["site"].tolist()
        else:
            ru4  = sorted(mdf4["rep"].unique())
            cm4r = plt.get_cmap("Set2")
            pal4 = {r: cm4r(i/max(len(ru4)-1, 1)) for i, r in enumerate(ru4)}
            gv4  = mdf4["rep"].tolist()

        fig_m, (ax_m, ax_ms) = plt.subplots(1, 2, figsize=(12, 6),
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
        ax_ms.axis("off")
        pl4 = set()
        for i, (xm, ym) in enumerate(co4):
            gv   = gv4[i]; col4 = pal4.get(gv, "#888")
            lb4  = mdf4.loc[i, "label"]; tr4 = mdf4.loc[i, "treatment"]
            mk4  = "o" if tr4 == "A" else "s"
            ax_m.scatter(xm, ym, color=col4, marker=mk4, s=110,
                         edgecolors="white", linewidths=0.8, zorder=3,
                         label=str(gv) if gv not in pl4 else "")
            ax_m.annotate(lb4, xy=(xm, ym), xytext=(6, 4), textcoords="offset points",
                          fontsize=8, color=col4, fontweight="bold")
            pl4.add(gv)
        if ell4:
            for gv, col4 in pal4.items():
                idx4 = [i for i, g in enumerate(gv4) if g == gv]
                if len(idx4) < 3: continue
                confidence_ellipse(co4[idx4, 0], co4[idx4, 1], ax_m,
                                   facecolor=col4, alpha=0.12, edgecolor=col4,
                                   linewidth=1.5, linestyle="--")
        ax_m.set_xlabel("NMDS axis 1", fontsize=11)
        ax_m.set_ylabel("NMDS axis 2", fontsize=11)
        ax_m.set_title(f"NMDS — Bray-Curtis  |  colour by {cb4}"
                       + ("  [reps combined]" if comb4 else ""),
                       fontsize=12, fontweight="bold")
        ax_m.axhline(0, color="#ccc", lw=0.8, zorder=1)
        ax_m.axvline(0, color="#ccc", lw=0.8, zorder=1)
        ax_m.grid(True, ls="--", alpha=0.25, zorder=0)
        ax_m.spines[["top", "right"]].set_visible(False)
        hm4 = [mpatches.Patch(color=c, label=str(g)) for g, c in pal4.items()]
        th4 = [ax_m.scatter([], [], marker="o", color="#555", s=60, label="Tr. A"),
               ax_m.scatter([], [], marker="s", color="#555", s=60, label="Tr. D")]
        ax_m.legend(handles=hm4+th4, title=f"Colour: {cb4}", fontsize=8, title_fontsize=8,
                    frameon=True, edgecolor="#ccc", loc="lower left")
        pc4 = "#c0392b" if (not np.isnan(p4) and p4 < 0.05) else "#27ae60"
        ps4 = f"{p4:.4f}" if not np.isnan(p4) else "N/A"
        fs4 = f"{f4:.3f}" if not np.isnan(f4) else "N/A"
        sg4 = ("p < 0.05\nCommunities differ\nsignificantly A vs D"
               if not np.isnan(p4) and p4 < 0.05
               else "p >= 0.05\nNo significant\ndifference detected")
        ax_ms.text(0.05, 0.97,
                   f"PERMANOVA\n(A vs D)\n{'─'*20}\nF : {fs4}\np : {ps4}\n"
                   f"Perms: {int(np4)}\n\n{'─'*20}\nStress: {st4:.4f}\n\n{'─'*20}\n"
                   f"< 0.05 excellent\n< 0.10 good\n< 0.20 ok\n> 0.20 poor",
                   transform=ax_ms.transAxes, va="top", ha="left", fontsize=9,
                   fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#ccc", lw=1))
        ax_ms.text(0.05, 0.28, sg4, transform=ax_ms.transAxes, va="top", ha="left",
                   fontsize=9, color=pc4, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=pc4, lw=1.5))
        fig_m.tight_layout()
        st.pyplot(fig_m, use_container_width=False)
        st.download_button("Download PNG", fig_to_bytes(fig_m),
                           file_name="nmds.png", mime="image/png")

    # ── Replicate similarity table ─────────────────────────────────────────
    elif plot_type2.startswith("Replicate"):
        c1, c2 = st.columns([2, 2])
        sc2 = c1.selectbox("Taxon level for BC", tc2_cands,
            index=tc2_cands.index(taxon_col2) if taxon_col2 in tc2_cands else 0, key="t2_sc")
        sg2 = c2.radio("Group by", ["site x treatment", "site only"], key="t2_sg")
        gk2 = ["_site", "_treatment", "_rep"] if sg2 == "site x treatment" else ["_site", "_rep"]
        lf2 = ((lambda k: f"{k[0]}{k[1]}{k[2]}") if sg2 == "site x treatment"
               else (lambda k: f"Site{k[0]}_Rep{k[1]}"))
        vecs2 = {}; labs5 = []
        for keys, sub in df2.groupby(gk2):
            keys = keys if isinstance(keys, tuple) else (keys,)
            lb5  = lf2(keys)
            grp  = sub.groupby(sc2)[reads_col2].sum()
            tot  = grp.sum()
            vecs2[lb5] = grp / tot * 100.0 if tot > 0 else grp * 0.0
            labs5.append(lb5)
        labs5 = sorted(labs5); n5 = len(labs5)
        at5   = sorted(set().union(*[v.index for v in vecs2.values()]))
        mat5  = np.zeros((n5, n5))
        for i, la in enumerate(labs5):
            for j, lb in enumerate(labs5):
                if i == j:
                    mat5[i, j] = 100.0
                elif i < j:
                    va5 = np.array([float(vecs2[la].get(t, 0)) for t in at5])
                    vb5 = np.array([float(vecs2[lb].get(t, 0)) for t in at5])
                    v5  = bc_sim(va5, vb5)
                    mat5[i, j] = v5 if not np.isnan(v5) else 0.0
                    mat5[j, i] = mat5[i, j]
        cell5 = max(0.55, min(1.2, 12.0/n5))
        fig_sim, ax_sim = plt.subplots(figsize=(max(8, n5*cell5+3), max(6, n5*cell5+2)))
        cbc5 = mcolors.LinearSegmentedColormap.from_list("bc", ["#c0392b", "#f39c12", "#27ae60"])
        im5  = ax_sim.imshow(mat5, cmap=cbc5, vmin=0, vmax=100, aspect="auto")
        fig_sim.colorbar(im5, ax=ax_sim, label="Bray-Curtis similarity (%)", fraction=0.03, pad=0.02)
        for i in range(n5):
            for j in range(n5):
                v5   = mat5[i, j]
                tc5  = "white" if (v5 < 35 or v5 > 80) else "black"
                ax_sim.text(j, i, f"{v5:.0f}", ha="center", va="center",
                            fontsize=max(6, min(10, 90//n5)), color=tc5, fontweight="bold")
        ax_sim.set_xticks(range(n5)); ax_sim.set_yticks(range(n5))
        ax_sim.set_xticklabels(labs5, rotation=45, ha="right", fontsize=max(6, min(9, 80//n5)))
        ax_sim.set_yticklabels(labs5, fontsize=max(6, min(9, 80//n5)))
        for tick, lb5 in zip(ax_sim.get_xticklabels(), labs5):
            if "A" in lb5: tick.set_color("#2c7bb6")
            elif "D" in lb5: tick.set_color("#d7191c")
        for tick, lb5 in zip(ax_sim.get_yticklabels(), labs5):
            if "A" in lb5: tick.set_color("#2c7bb6")
            elif "D" in lb5: tick.set_color("#d7191c")
        if sg2 == "site x treatment":
            prev5 = None
            for j, lb5 in enumerate(labs5):
                m5 = re.match(r"^(\d+)", lb5); s5 = m5.group(1) if m5 else None
                if s5 != prev5 and j > 0:
                    ax_sim.axhline(j-0.5, color="white", lw=2)
                    ax_sim.axvline(j-0.5, color="white", lw=2)
                prev5 = s5
        ax_sim.set_title(f"Replicate BC similarity matrix — {sc2}",
                         fontsize=13, fontweight="bold", pad=12)
        ax_sim.set_xlabel("Sample", fontsize=11); ax_sim.set_ylabel("Sample", fontsize=11)
        ax_sim.legend(handles=[mpatches.Patch(color="#2c7bb6", label="Treatment A"),
                                mpatches.Patch(color="#d7191c", label="Treatment D")],
                      fontsize=9, loc="upper right", bbox_to_anchor=(1.18, 1.12),
                      frameon=True, edgecolor="#ccc")
        fig_sim.tight_layout()
        st.pyplot(fig_sim, use_container_width=False)
        st.download_button("Download PNG", fig_to_bytes(fig_sim),
                           file_name="similarity_table.png", mime="image/png")
        st.subheader("Similarity values (table)")
        sdf5 = pd.DataFrame(mat5, index=labs5, columns=labs5).round(1)
        st.dataframe(sdf5.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                     use_container_width=True)
