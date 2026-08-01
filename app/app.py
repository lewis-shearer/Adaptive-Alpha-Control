"""
Adaptive Alpha Control — Results Dashboard
============================================
A Streamlit dashboard for evaluating "Adaptive Alpha Control": a trend-aware
dynamic weighting mechanism for stable adversarial debiasing, extending
Zhang et al. (2018).

Compares:
  - Baseline (no debiasing)
  - 8 Fixed-alpha adversarial debiasing runs (Zhang et al. 2018 reproduction)
  - Dynamic Alpha (Adaptive Alpha Control)
...each run repeated across 30 random seeds, logged via Weights & Biases.

Run with:
    streamlit run app.py

Data:
    Drop your W&B export CSV in the same folder as this script (it will be
    auto-detected), or upload it via the sidebar. The app expects the export
    to contain (at minimum) these columns: Name, seed, alpha, ACC, DAO, DEO.
    Run names are expected to follow the convention used in the
    Adaptive-Alpha-Control repo, e.g.:
        baseline_adult_gender_seed0
        fixed_alpha0.3_adult_gender_seed0
        dynamic_adult_gender_seed0
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Adaptive Alpha Control Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASELINE_LABEL = "Baseline"
DYNAMIC_LABEL = "Dynamic (Adaptive Alpha)"
ALPHA_INIT = 0.4708055927870487  # dynamic controller's alpha_init, used to label the matching fixed run

NAME_RE = re.compile(
    r"^(baseline|dynamic|fixed_alpha([0-9.]+))_.*seed(\d+)$", re.IGNORECASE
)

METRIC_INFO = {
    "ACC": {"label": "Accuracy (ACC)", "direction": "higher is better"},
    "DEO": {"label": "Difference in Equal Opportunity (DEO)", "direction": "lower is better"},
    "DAO": {"label": "Difference in Adversarial Outcomes (DAO)", "direction": "lower is better"},
}


# --------------------------------------------------------------------------
# Parsing & loading
# --------------------------------------------------------------------------

def parse_run_name(name: str):
    """Return (group_label, alpha_value_or_nan, seed) parsed from a run Name."""
    if not isinstance(name, str):
        return None, np.nan, np.nan
    m = NAME_RE.match(name.strip())
    if not m:
        return None, np.nan, np.nan
    kind, alpha_str, seed = m.groups()
    seed = int(seed)
    kind = kind.lower()
    if kind == "baseline":
        return BASELINE_LABEL, np.nan, seed
    if kind == "dynamic":
        return DYNAMIC_LABEL, np.nan, seed
    alpha_val = float(alpha_str)
    if abs(alpha_val - ALPHA_INIT) < 1e-4:
        label = f"Fixed α={alpha_val:.4f} (= dynamic init)"
    else:
        label = f"Fixed α={alpha_val:g}"
    return label, alpha_val, seed


def build_group_order(groups: list[str]) -> list[str]:
    """Order: Baseline, fixed alphas ascending, Dynamic last."""
    fixed = [g for g in groups if g not in (BASELINE_LABEL, DYNAMIC_LABEL)]

    def alpha_of(g):
        m = re.search(r"([0-9.]+)", g)
        return float(m.group(1)) if m else 0.0

    fixed_sorted = sorted(fixed, key=alpha_of)
    order = []
    if BASELINE_LABEL in groups:
        order.append(BASELINE_LABEL)
    order += fixed_sorted
    if DYNAMIC_LABEL in groups:
        order.append(DYNAMIC_LABEL)
    return order


@st.cache_data(show_spinner=False)
def load_data(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip() for c in df.columns]

    if "Name" not in df.columns:
        raise ValueError("CSV has no 'Name' column — is this a W&B run export?")

    parsed = df["Name"].apply(parse_run_name)
    df["group"] = [p[0] for p in parsed]
    df["alpha_fixed_value"] = [p[1] for p in parsed]
    df["seed_parsed"] = [p[2] for p in parsed]

    # Prefer the parsed seed, fall back to the CSV's own seed column
    if "seed" in df.columns:
        df["seed_final"] = df["seed_parsed"].fillna(pd.to_numeric(df["seed"], errors="coerce"))
    else:
        df["seed_final"] = df["seed_parsed"]

    df["alpha_used"] = pd.to_numeric(df.get("alpha"), errors="coerce")

    for col in ["ACC", "DAO", "DEO"]:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required metric column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["pressure", "ema_dacc", "ema_ddao", "ema_ddeo", "pred_loss", "adv_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["group", "ACC", "DEO", "DAO"]).reset_index(drop=True)
    return df


def is_pareto_efficient(acc: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Points that maximize acc and minimize cost. Returns boolean mask of
    non-dominated (Pareto-efficient) points."""
    n = len(acc)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            dominates = (acc[j] >= acc[i]) and (cost[j] <= cost[i]) and (
                acc[j] > acc[i] or cost[j] < cost[i]
            )
            if dominates:
                efficient[i] = False
                break
    return efficient


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("group").agg(
        n_runs=("ACC", "count"),
        acc_mean=("ACC", "mean"), acc_std=("ACC", "std"),
        deo_mean=("DEO", "mean"), deo_std=("DEO", "std"),
        dao_mean=("DAO", "mean"), dao_std=("DAO", "std"),
        alpha_mean=("alpha_used", "mean"),
    ).reset_index()
    return agg


def compare_groups(df: pd.DataFrame, group_a: str, group_b: str, metric: str) -> dict:
    """Compare two groups on a metric. Uses a paired test (matched by seed) if
    every seed present in both groups appears exactly once in each — this is
    the case whenever the sweep reuses the same 30 seeds across conditions,
    and is far more powerful than an unpaired test since it removes
    seed-to-seed (init / train-test split) variance as a confound. Falls back
    to an unpaired test otherwise."""
    a = df[df["group"] == group_a][["seed_final", metric]].dropna()
    b = df[df["group"] == group_b][["seed_final", metric]].dropna()

    can_pair = (
        not a.empty and not b.empty
        and not a["seed_final"].duplicated().any()
        and not b["seed_final"].duplicated().any()
    )
    merged = a.merge(b, on="seed_final", suffixes=("_a", "_b")) if can_pair else pd.DataFrame()

    if can_pair and len(merged) >= 8:
        diff = merged[f"{metric}_a"] - merged[f"{metric}_b"]
        try:
            _, w_p = scipy_stats.wilcoxon(diff)
        except ValueError:
            w_p = np.nan
        try:
            _, t_p = scipy_stats.ttest_rel(merged[f"{metric}_a"], merged[f"{metric}_b"])
        except ValueError:
            t_p = np.nan
        sd = diff.std(ddof=1)
        effect = diff.mean() / sd if sd else np.nan
        return dict(
            n=len(merged), paired=True, mean_diff=diff.mean(),
            p_value=w_p, p_test="Wilcoxon signed-rank (paired)",
            p_value_alt=t_p, effect_size=effect, effect_label="Cohen's dz (paired)",
        )

    a_vals, b_vals = a[metric], b[metric]
    if a_vals.empty or b_vals.empty:
        return dict(n=0, paired=False, mean_diff=np.nan, p_value=np.nan,
                     p_test="n/a", p_value_alt=np.nan, effect_size=np.nan, effect_label="n/a")
    try:
        _, w_p = scipy_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    except ValueError:
        w_p = np.nan
    _, t_p = scipy_stats.ttest_ind(a_vals, b_vals, equal_var=False)
    pooled_sd = np.sqrt((a_vals.std(ddof=1) ** 2 + b_vals.std(ddof=1) ** 2) / 2)
    effect = (a_vals.mean() - b_vals.mean()) / pooled_sd if pooled_sd else np.nan
    return dict(
        n=min(len(a_vals), len(b_vals)), paired=False, mean_diff=a_vals.mean() - b_vals.mean(),
        p_value=w_p, p_test="Mann-Whitney U (unpaired)",
        p_value_alt=t_p, effect_size=effect, effect_label="Cohen's d (unpaired)",
    )


def build_significance_table(df: pd.DataFrame, metric: str, reference_group: str, other_groups: list[str]) -> pd.DataFrame:
    rows = []
    for g in other_groups:
        if g == reference_group:
            continue
        res = compare_groups(df, reference_group, g, metric)
        rows.append({
            "Comparison": f"{reference_group} vs {g}",
            "n pairs/obs": res["n"],
            "Test": res["p_test"],
            "Mean diff": res["mean_diff"],
            "p-value (raw)": res["p_value"],
            "Effect size": res["effect_size"],
            "Effect metric": res["effect_label"],
        })
    out = pd.DataFrame(rows)
    if not out.empty and out["p-value (raw)"].notna().any():
        mask = out["p-value (raw)"].notna()
        corrected = np.full(len(out), np.nan)
        reject, p_holm, _, _ = multipletests(out.loc[mask, "p-value (raw)"], method="holm")
        corrected[mask.to_numpy()] = p_holm
        out["p-value (Holm-corrected)"] = corrected
        out["Significant (α=0.05, corrected)"] = out["p-value (Holm-corrected)"] < 0.05
    return out


# --------------------------------------------------------------------------
# Sidebar — data loading
# --------------------------------------------------------------------------

st.sidebar.title("\U0001F4CA Data")
uploaded = st.sidebar.file_uploader("Upload W&B export CSV", type=["csv"])

df = None
source_note = None

if uploaded is not None:
    try:
        df = load_data(uploaded)
        source_note = f"Loaded upload: **{uploaded.name}**"
    except Exception as e:
        st.sidebar.error(f"Could not parse uploaded file: {e}")

if df is None:
    here = Path(__file__).resolve().parent
    candidates = sorted(here.glob("*.csv"))
    if candidates:
        try:
            df = load_data(candidates[0])
            source_note = f"Auto-loaded local file: **{candidates[0].name}**"
        except Exception as e:
            st.sidebar.warning(f"Found {candidates[0].name} but couldn't parse it: {e}")

if df is None:
    st.title("Adaptive Alpha Control — Results Dashboard")
    st.info(
        "No data loaded yet. Upload your Weights & Biases export CSV using the "
        "sidebar, or place it in the same folder as this script and rerun."
    )
    st.stop()

st.sidebar.success(source_note)
st.sidebar.caption(f"{len(df)} runs loaded across {df['group'].nunique()} groups.")

GROUP_ORDER = build_group_order(df["group"].unique().tolist())
palette = px.colors.sequential.Blues[2:]
COLOR_MAP = {}
fixed_groups = [g for g in GROUP_ORDER if g not in (BASELINE_LABEL, DYNAMIC_LABEL)]
for i, g in enumerate(fixed_groups):
    COLOR_MAP[g] = palette[min(i, len(palette) - 1)]
COLOR_MAP[BASELINE_LABEL] = "#888888"
COLOR_MAP[DYNAMIC_LABEL] = "#e6550d"

group_filter = st.sidebar.multiselect(
    "Groups to include", GROUP_ORDER, default=GROUP_ORDER
)
df = df[df["group"].isin(group_filter)]
GROUP_ORDER = [g for g in GROUP_ORDER if g in group_filter]

stats = summarize(df)
stats["group"] = pd.Categorical(stats["group"], categories=GROUP_ORDER, ordered=True)
stats = stats.sort_values("group")

baseline_row = stats[stats["group"] == BASELINE_LABEL]
dynamic_row = stats[stats["group"] == DYNAMIC_LABEL]
fixed_rows = stats[~stats["group"].isin([BASELINE_LABEL, DYNAMIC_LABEL])]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Adaptive Alpha Control — Results Dashboard")
st.caption(
    "Trend-aware dynamic weighting for stable adversarial debiasing — "
    "an extension of Zhang et al. (2018) — evaluated against a no-debiasing "
    "baseline and 8 fixed-alpha configurations, 30 seeds per condition."
)

kpi_cols = st.columns(4)
if not dynamic_row.empty and not baseline_row.empty:
    d, b = dynamic_row.iloc[0], baseline_row.iloc[0]
    kpi_cols[0].metric("Dynamic ACC", f"{d['acc_mean']:.4f}", f"{d['acc_mean'] - b['acc_mean']:+.4f} vs baseline")
    kpi_cols[1].metric("Dynamic DEO", f"{d['deo_mean']:.4f}", f"{d['deo_mean'] - b['deo_mean']:+.4f} vs baseline", delta_color="inverse")
    kpi_cols[2].metric("Dynamic DAO", f"{d['dao_mean']:.4f}", f"{d['dao_mean'] - b['dao_mean']:+.4f} vs baseline", delta_color="inverse")
    if not fixed_rows.empty:
        best_fixed_deo = fixed_rows.loc[fixed_rows["deo_std"].idxmin()]
        kpi_cols[3].metric(
            "DEO std: Dynamic vs best fixed",
            f"{d['deo_std']:.4f}",
            f"{d['deo_std'] - best_fixed_deo['deo_std']:+.4f}",
            delta_color="inverse",
        )
else:
    for c, label in zip(kpi_cols, ["ACC", "DEO", "DAO", "n runs"]):
        c.metric(label, "—")

st.divider()

tab_tradeoff, tab_stability, tab_alpha, tab_verdict, tab_method, tab_data = st.tabs(
    ["⚖️ Fairness–Accuracy Tradeoff", "\U0001F4C8 Stability Across Seeds",
     "\U0001F39B️ Alpha Dynamics", "✅ Verdict", "\U0001F52C Methodology", "\U0001F4C4 Raw Data"]
)

# --------------------------------------------------------------------------
# Tab 1: Fairness-accuracy tradeoff / Pareto frontier
# --------------------------------------------------------------------------

with tab_tradeoff:
    st.subheader("Does Dynamic Alpha beat the fixed-alpha tradeoff curve?")
    st.markdown(
        "Each point is the mean of 30 seeds for one configuration. Error bars show "
        "±1 standard deviation. The dashed line traces the fixed-alpha sweep in "
        "increasing order of α — this is the tradeoff curve a practitioner gets "
        "from Zhang et al. (2018) alone. A point that sits **up and to the right** of "
        "this curve is a Pareto improvement: better fairness *and* accuracy."
    )

    metric_choice = st.radio("Fairness metric", ["DEO", "DAO"], horizontal=True, key="tradeoff_metric")
    mean_col, std_col = f"{metric_choice.lower()}_mean", f"{metric_choice.lower()}_std"

    plot_df = stats.dropna(subset=["acc_mean", mean_col]).copy()
    plot_df["is_dynamic"] = plot_df["group"] == DYNAMIC_LABEL
    plot_df["is_baseline"] = plot_df["group"] == BASELINE_LABEL

    pareto_mask = is_pareto_efficient(plot_df["acc_mean"].to_numpy(), plot_df[mean_col].to_numpy())
    plot_df["pareto"] = pareto_mask

    fig = go.Figure()

    # fixed-alpha tradeoff line
    fixed_only = plot_df[~plot_df["is_dynamic"] & ~plot_df["is_baseline"]].sort_values("alpha_mean")
    if not fixed_only.empty:
        fig.add_trace(go.Scatter(
            x=fixed_only["acc_mean"], y=fixed_only[mean_col],
            mode="lines+markers",
            line=dict(color="#4292c6", dash="dash"),
            marker=dict(size=10, color="#4292c6"),
            name="Fixed α sweep",
            error_x=dict(type="data", array=fixed_only["acc_std"], visible=True),
            error_y=dict(type="data", array=fixed_only[std_col], visible=True),
            text=fixed_only["group"],
            hovertemplate="%{text}<br>ACC=%{x:.4f}<br>" + metric_choice + "=%{y:.4f}<extra></extra>",
        ))

    if not plot_df[plot_df["is_baseline"]].empty:
        b = plot_df[plot_df["is_baseline"]].iloc[0]
        fig.add_trace(go.Scatter(
            x=[b["acc_mean"]], y=[b[mean_col]], mode="markers",
            marker=dict(size=16, symbol="square", color="#888888"),
            name=BASELINE_LABEL,
            error_x=dict(type="data", array=[b["acc_std"]], visible=True),
            error_y=dict(type="data", array=[b[std_col]], visible=True),
            hovertemplate=f"{BASELINE_LABEL}<br>ACC=%{{x:.4f}}<br>{metric_choice}=%{{y:.4f}}<extra></extra>",
        ))

    if not plot_df[plot_df["is_dynamic"]].empty:
        d = plot_df[plot_df["is_dynamic"]].iloc[0]
        fig.add_trace(go.Scatter(
            x=[d["acc_mean"]], y=[d[mean_col]], mode="markers",
            marker=dict(size=20, symbol="star", color="#e6550d", line=dict(width=1, color="black")),
            name=DYNAMIC_LABEL,
            error_x=dict(type="data", array=[d["acc_std"]], visible=True),
            error_y=dict(type="data", array=[d[std_col]], visible=True),
            hovertemplate=f"{DYNAMIC_LABEL}<br>ACC=%{{x:.4f}}<br>{metric_choice}=%{{y:.4f}}<extra></extra>",
        ))

    fig.add_vline(x=0.84, line_dash="dot", line_color="gray", annotation_text="accuracy floor (0.84)")
    fig.update_layout(
        xaxis_title="Mean Accuracy (higher → better, right)",
        yaxis_title=f"Mean {metric_choice} (lower → better, down)",
        yaxis=dict(autorange="reversed"),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: the y-axis is reversed so that 'better' fairness (lower "
        f"{metric_choice}) points appear higher on the chart, matching accuracy's "
        "'higher is better' orientation — up-and-right is always the improvement direction."
    )

    if not dynamic_row.empty:
        dyn_on_frontier = plot_df.loc[plot_df["is_dynamic"], "pareto"]
        if not dyn_on_frontier.empty and dyn_on_frontier.iloc[0]:
            st.success(
                f"Dynamic Alpha is **Pareto-efficient** on {metric_choice}: no fixed-alpha "
                "run (or baseline) beats it on both accuracy and fairness simultaneously."
            )
        elif not dyn_on_frontier.empty:
            dominators = plot_df[plot_df["pareto"] & (plot_df["acc_mean"] >= plot_df.loc[plot_df["is_dynamic"], "acc_mean"].iloc[0])]
            st.warning(
                f"Dynamic Alpha is **dominated** on {metric_choice} by: "
                + ", ".join(dominators["group"].tolist())
            )

# --------------------------------------------------------------------------
# Tab 2: Stability across seeds
# --------------------------------------------------------------------------

with tab_stability:
    st.subheader("How consistent is each mechanism across 30 seeds?")
    st.markdown(
        "A debiasing method isn't just judged on its *average* fairness — a method "
        "that's great on some seeds and terrible on others is unreliable in practice. "
        "This view compares the **spread** of outcomes, not just the mean."
    )

    metric_choice2 = st.radio("Metric", ["ACC", "DEO", "DAO"], horizontal=True, key="stability_metric")

    box_df = df.copy()
    box_df["group"] = pd.Categorical(box_df["group"], categories=GROUP_ORDER, ordered=True)
    box_df = box_df.sort_values("group")

    fig_box = px.box(
        box_df, x="group", y=metric_choice2, color="group",
        color_discrete_map=COLOR_MAP, points="all",
        category_orders={"group": GROUP_ORDER},
    )
    fig_box.update_layout(
        showlegend=False, height=520,
        xaxis_title="", yaxis_title=METRIC_INFO[metric_choice2]["label"],
    )
    st.plotly_chart(fig_box, use_container_width=True)

    std_df = stats.copy()
    std_df["group"] = pd.Categorical(std_df["group"], categories=GROUP_ORDER, ordered=True)
    std_df = std_df.sort_values("group")
    fig_std = px.bar(
        std_df, x="group", y=f"{metric_choice2.lower()}_std", color="group",
        color_discrete_map=COLOR_MAP, category_orders={"group": GROUP_ORDER},
    )
    fig_std.update_layout(
        showlegend=False, height=380,
        xaxis_title="", yaxis_title=f"Std. dev. of {metric_choice2} across seeds",
        title="Lower bars = more stable / reproducible across seeds",
    )
    st.plotly_chart(fig_std, use_container_width=True)

    st.markdown("#### Verdict: where does Dynamic Alpha rank on stability?")
    std_col2 = f"{metric_choice2.lower()}_std"
    rank_df = std_df[["group", std_col2]].dropna().sort_values(std_col2).reset_index(drop=True)
    rank_df.insert(0, "rank", rank_df.index + 1)
    n_groups = len(rank_df)

    dyn_rank_row = rank_df[rank_df["group"] == DYNAMIC_LABEL]
    if dyn_rank_row.empty or n_groups < 2:
        st.info("Need Dynamic Alpha plus at least one other group loaded to rank stability.")
    else:
        dyn_rank = int(dyn_rank_row["rank"].iloc[0])
        dyn_std_val = float(dyn_rank_row[std_col2].iloc[0])
        more_stable_than = rank_df[rank_df["rank"] > dyn_rank]["group"].tolist()
        less_stable_than = rank_df[rank_df["rank"] < dyn_rank]["group"].tolist()
        # percentile: 100% = most stable of all groups, 0% = least stable
        percentile = 100 * (n_groups - dyn_rank) / (n_groups - 1)

        with st.expander("Full stability ranking (lower std = more stable)", expanded=False):
            st.dataframe(rank_df.rename(columns={std_col2: f"{metric_choice2} std"}),
                         use_container_width=True, hide_index=True)

        if dyn_rank == 1:
            st.success(
                f"**Most stable of all {n_groups} groups on {metric_choice2}** "
                f"(std = {dyn_std_val:.4f}). No configuration — fixed-alpha or baseline — "
                "produces more consistent outcomes across seeds."
            )
        elif percentile >= 70:
            st.success(
                f"**High end of stability on {metric_choice2}** — ranks {dyn_rank} of {n_groups} "
                f"(std = {dyn_std_val:.4f}, {percentile:.0f}th percentile). More stable than: "
                f"{', '.join(more_stable_than) if more_stable_than else 'none'}."
            )
        elif percentile >= 40:
            st.info(
                f"**Mid-pack stability on {metric_choice2}** — ranks {dyn_rank} of {n_groups} "
                f"(std = {dyn_std_val:.4f}, {percentile:.0f}th percentile). More stable than "
                f"{', '.join(more_stable_than) if more_stable_than else 'none'}; less stable than "
                f"{', '.join(less_stable_than) if less_stable_than else 'none'}."
            )
        else:
            st.warning(
                f"**Low end of stability on {metric_choice2}** — ranks {dyn_rank} of {n_groups} "
                f"(std = {dyn_std_val:.4f}, {percentile:.0f}th percentile), less stable than "
                f"{len(less_stable_than)} other group(s): {', '.join(less_stable_than)}. "
                "If this holds for DEO and DAO both, it's worth checking whether the controller's "
                "pressure term is overreacting on certain seeds (see Alpha Dynamics tab)."
            )

        if not fixed_rows.empty:
            avg_fixed_std = fixed_rows[std_col2].mean()
            pct_vs_avg = 100 * (avg_fixed_std - dyn_std_val) / avg_fixed_std if avg_fixed_std else 0
            st.caption(
                f"For reference: Dynamic Alpha's std. dev. is {pct_vs_avg:+.1f}% relative to the "
                f"*average* of the 8 fixed-alpha runs ({dyn_std_val:.4f} vs {avg_fixed_std:.4f}). "
                "Positive = more stable than the average fixed-alpha configuration."
            )

# --------------------------------------------------------------------------
# Tab 3: Alpha dynamics (dynamic runs only)
# --------------------------------------------------------------------------

with tab_alpha:
    st.subheader("What does the controller actually do?")
    st.markdown(
        "Unlike a fixed-alpha run, Dynamic Alpha doesn't have one α to report — it "
        "starts at `α_init` and adjusts every epoch based on the *pressure signal* "
        "(a weighted blend of the smoothed changes in accuracy, DEO, and DAO). "
        "These charts look at where each of the 30 seeds' controllers ended up, "
        "and whether that end point actually explains the fairness outcome."
    )
    dyn_df = df[df["group"] == DYNAMIC_LABEL].copy()
    if dyn_df.empty:
        st.info("No Dynamic Alpha runs found in the loaded data.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_alpha_hist = px.histogram(
                dyn_df, x="alpha_used", nbins=15,
                title="Distribution of converged final α (across 30 seeds)",
            )
            fig_alpha_hist.add_vline(x=ALPHA_INIT, line_dash="dot", line_color="gray",
                                      annotation_text="α_init")
            fig_alpha_hist.update_layout(xaxis_title="Final α", yaxis_title="Count of seeds")
            st.plotly_chart(fig_alpha_hist, use_container_width=True)
            alpha_cv = dyn_df["alpha_used"].std() / dyn_df["alpha_used"].mean() if dyn_df["alpha_used"].mean() else np.nan
            st.caption(
                "**What this shows:** every seed starts at the same `α_init` (dotted line) "
                "but the controller pushes it up or down depending on what happens during "
                "training for that particular seed. A **narrow** distribution centred near "
                "`α_init` means the controller reliably settles near the same operating point "
                "regardless of random initialization — a sign of a stable control loop. A "
                "**wide or multimodal** distribution means the controller's trajectory is "
                f"sensitive to seed noise (coefficient of variation here: {alpha_cv:.2f}). "
                "Neither is automatically 'better' — a wide spread could mean the controller "
                "is correctly adapting to genuinely harder seeds, or it could mean it's unstable; "
                "the scatter plot on the right helps distinguish the two."
            )
        with c2:
            metric_for_scatter = st.selectbox("Plot final α against", ["DEO", "DAO", "ACC"])
            fig_scatter = px.scatter(
                dyn_df, x="alpha_used", y=metric_for_scatter, trendline="ols",
                hover_data=["seed_final"],
                title=f"Final α vs {metric_for_scatter} (one point per seed)",
            )
            fig_scatter.update_layout(xaxis_title="Final α", yaxis_title=metric_for_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
            corr = dyn_df[["alpha_used", metric_for_scatter]].corr().iloc[0, 1]
            direction = "higher final α → lower (better) " if corr < 0 else "higher final α → higher "
            st.caption(
                f"**What this shows:** each dot is one seed's final converged α plotted against "
                f"the {metric_for_scatter} it ended up with; the line is a linear (OLS) fit. "
                f"Correlation here is **r = {corr:.2f}**. If the trend is clearly downward for "
                "DEO/DAO, that's evidence the controller is doing something meaningful — seeds "
                "where it pushed α higher genuinely ended up fairer, i.e. the endpoint of the "
                "trajectory explains the outcome. A flat or noisy trend means the *final* α "
                "isn't the main story — the path it took to get there (captured by the internals "
                "below) may matter more than where it landed."
            )

        extra_cols = [c for c in ["pressure", "ema_dacc", "ema_ddao", "ema_ddeo"] if c in dyn_df.columns]
        if extra_cols:
            st.markdown("**Final-epoch controller internals** (across seeds)")
            fig_extra = px.box(dyn_df.melt(value_vars=extra_cols, var_name="signal", value_name="value"),
                                x="signal", y="value", points="all")
            fig_extra.update_layout(height=420, xaxis_title="", yaxis_title="Value at final epoch")
            st.plotly_chart(fig_extra, use_container_width=True)
            st.caption(
                "**What this shows:** the internal signals the controller was computing at the "
                "*last* epoch of training, one point per seed. `pressure` is the combined, "
                "weighted push on α for the next step (positive = increase adversarial weight, "
                "negative = decrease it) — if these cluster tightly around zero at epoch 30, the "
                "controller has stabilized rather than still oscillating when training stopped. "
                "`ema_dacc` / `ema_ddao` / `ema_ddeo` are the exponentially-smoothed epoch-to-epoch "
                "changes in accuracy, DAO, and DEO respectively that feed into that pressure "
                "calculation — large remaining spread in these at the final epoch suggests some "
                "seeds hadn't fully converged within the 30-epoch budget, which would be worth "
                "checking against per-epoch logs (this dashboard only has the final-epoch snapshot)."
            )
        st.caption(
            "Overall: the controller doesn't converge on one universal α — it adapts per run. "
            "That's the whole premise of Adaptive Alpha Control, so seed-to-seed variation in "
            "final α is expected; the question is whether that variation is *doing something "
            "useful* (correlating with better fairness) or just adding noise."
        )

# --------------------------------------------------------------------------
# Tab 4: Verdict
# --------------------------------------------------------------------------

with tab_verdict:
    st.subheader("Summary table")
    display_stats = stats.copy()
    display_stats = display_stats[[
        "group", "n_runs", "acc_mean", "acc_std", "deo_mean", "deo_std", "dao_mean", "dao_std", "alpha_mean"
    ]]
    display_stats.columns = [
        "Group", "# runs", "ACC (mean)", "ACC (std)", "DEO (mean)", "DEO (std)",
        "DAO (mean)", "DAO (std)", "Mean α used",
    ]
    st.dataframe(
        display_stats.style.format({
            "ACC (mean)": "{:.4f}", "ACC (std)": "{:.4f}",
            "DEO (mean)": "{:.4f}", "DEO (std)": "{:.4f}",
            "DAO (mean)": "{:.4f}", "DAO (std)": "{:.4f}",
            "Mean α used": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Automated read-out")
    if dynamic_row.empty or baseline_row.empty or fixed_rows.empty:
        st.info("Need Baseline, at least one Fixed α run, and Dynamic Alpha loaded to generate a verdict.")
    else:
        d = dynamic_row.iloc[0]
        b = baseline_row.iloc[0]
        best_deo_fixed = fixed_rows.loc[fixed_rows["deo_mean"].idxmin()]
        best_dao_fixed = fixed_rows.loc[fixed_rows["dao_mean"].idxmin()]

        acc_delta_vs_base = d["acc_mean"] - b["acc_mean"]
        deo_reduction_vs_base = 100 * (b["deo_mean"] - d["deo_mean"]) / b["deo_mean"]
        dao_reduction_vs_base = 100 * (b["dao_mean"] - d["dao_mean"]) / b["dao_mean"]
        deo_reduction_vs_best_fixed = 100 * (best_deo_fixed["deo_mean"] - d["deo_mean"]) / best_deo_fixed["deo_mean"]
        dao_reduction_vs_best_fixed = 100 * (best_dao_fixed["dao_mean"] - d["dao_mean"]) / best_dao_fixed["dao_mean"]

        lines = []
        lines.append(
            f"- **Accuracy:** Dynamic Alpha changes mean accuracy by **{acc_delta_vs_base:+.4f}** "
            f"vs baseline ({d['acc_mean']:.4f} vs {b['acc_mean']:.4f}) — "
            + ("essentially preserved." if abs(acc_delta_vs_base) < 0.005 else "a meaningful shift.")
        )
        lines.append(
            f"- **DEO vs baseline:** {deo_reduction_vs_base:+.1f}% "
            f"({d['deo_mean']:.4f} vs {b['deo_mean']:.4f})."
        )
        lines.append(
            f"- **DAO vs baseline:** {dao_reduction_vs_base:+.1f}% "
            f"({d['dao_mean']:.4f} vs {b['dao_mean']:.4f})."
        )
        lines.append(
            f"- **DEO vs best fixed-α run** ({best_deo_fixed['group']}): "
            f"{deo_reduction_vs_best_fixed:+.1f}% ({d['deo_mean']:.4f} vs {best_deo_fixed['deo_mean']:.4f})."
        )
        lines.append(
            f"- **DAO vs best fixed-α run** ({best_dao_fixed['group']}): "
            f"{dao_reduction_vs_best_fixed:+.1f}% ({d['dao_mean']:.4f} vs {best_dao_fixed['dao_mean']:.4f})."
        )
        avg_fixed_deo_std = fixed_rows["deo_std"].mean()
        avg_fixed_dao_std = fixed_rows["dao_std"].mean()
        deo_std_pct = 100 * (avg_fixed_deo_std - d["deo_std"]) / avg_fixed_deo_std
        dao_std_pct = 100 * (avg_fixed_dao_std - d["dao_std"]) / avg_fixed_dao_std
        lines.append(
            f"- **Stability:** DEO std. dev. is {deo_std_pct:+.1f}% vs the average fixed-α run; "
            f"DAO std. dev. is {dao_std_pct:+.1f}% (positive = Dynamic is more stable)."
        )
        st.markdown("\n".join(lines))

        working = (deo_reduction_vs_base > 0 and dao_reduction_vs_base > 0 and abs(acc_delta_vs_base) < 0.01)
        if working:
            st.success(
                "**Overall: the mechanism appears to be working as intended** — it reduces both "
                "fairness metrics relative to baseline while keeping accuracy within a small margin, "
                "without requiring manual alpha tuning."
            )
        else:
            st.warning(
                "**Overall: mixed evidence.** Check the numbers above — either the accuracy cost is "
                "larger than expected, or fairness didn't improve on both metrics. Inspect the "
                "tradeoff and stability tabs before drawing conclusions."
            )

# --------------------------------------------------------------------------
# Tab 5: Methodology & experimental soundness
# --------------------------------------------------------------------------

with tab_method:
    st.subheader("Is this experimental design sound?")
    st.markdown(
        "Short answer: **the design is reasonable, but the point estimates on the "
        "other tabs are not, by themselves, a statistical claim that Dynamic Alpha "
        "is 'better.'** Below is what the design gets right, what would need to be "
        "true (or added) before publishing a stronger claim, and a live significance "
        "test computed from your actual data."
    )

    st.markdown("#### What's methodologically solid")
    st.markdown(
        "- **A real baseline and a faithful prior-method reproduction**, not a straw "
        "man — you're comparing against Zhang et al. (2018)'s own mechanism, run "
        "properly, not a weakened version of it.\n"
        "- **A full fixed-alpha sweep (8 values), not one cherry-picked comparison "
        "point.** This is important: if you'd only run one fixed α and compared it "
        "to Dynamic, any result would be an artifact of which α you happened to pick. "
        "Tracing the whole tradeoff curve (as the Tradeoff tab does) and checking "
        "Pareto-dominance across *all* of it is the right way to use a sweep like this.\n"
        "- **30 seeds per condition** is enough to estimate variance and run "
        "reasonably powered statistical tests — most papers in this space use 5–10.\n"
        "- **Identical training budget/architecture across conditions** (30 epochs, "
        "batch size 256, same predictor/adversary architecture) — differences aren't "
        "confounded by unequal compute."
    )

    st.markdown("#### What to check before claiming 'better'")
    st.markdown(
        "**1. Statistical significance.** A mean difference of, say, 0.02 in DEO means "
        "nothing on its own without knowing whether that gap is larger than what 30 "
        "seeds' worth of noise could produce by chance. The table below runs an actual "
        "test on your data for this.\n\n"
        "**2. Multiple comparisons / 'best-of-8' selection bias.** If you pick whichever "
        "fixed α happens to score best on DEO in *this* sample and compare Dynamic only "
        "against that one, you're comparing Dynamic against a winner selected "
        "post-hoc — which is optimistic for the fixed-alpha side (regression to the "
        "mean / winner's-curse effect) and also invalidates a single p-value unless "
        "you correct for having tested 8 candidates. This dashboard's Holm-corrected "
        "test below already accounts for that; the Tradeoff tab's Pareto check "
        "(which uses the *whole* curve, not one winner) avoids the problem entirely — "
        "prefer that framing over 'Dynamic vs the best fixed α.'\n\n"
        "**3. Paired vs unpaired seeds.** If seed `k` uses the same train/test split "
        "and weight initialization across every condition (looks likely here, since "
        "seed ranges 0–29 identically in every group), you should analyze it as a "
        "**paired** design — comparing seed 5's baseline run to seed 5's dynamic run, "
        "not treating the 30 baseline runs and 30 dynamic runs as independent samples. "
        "Paired tests have much more power because they cancel out seed-specific noise "
        "(data split luck, init luck). The test below auto-detects this and uses "
        "Wilcoxon signed-rank when it can.\n\n"
        "**4. Was the controller tuned harder than the baseline it's compared to?** "
        "Your dynamic controller's hyperparameters (`alpha_lr = 0.4468379573`, "
        "`w_acc = 4.6453214453`, etc.) have the precision you'd expect from an "
        "automated search (e.g. Optuna/Bayesian optimization), while the fixed-alpha "
        "values are round numbers (0.05, 0.1, 0.2 … 1.5). If the controller's "
        "hyperparameters were tuned with a search budget that the fixed-alpha baseline "
        "never got, part of Dynamic's advantage could simply be 'we tuned this one and "
        "not the other,' rather than the trend-aware mechanism itself. Worth stating "
        "explicitly in the write-up how each was tuned (or wasn't), and ideally "
        "reporting a sensitivity analysis showing Dynamic still wins across a range of "
        "its own hyperparameters, not just the single best setting found.\n\n"
        "**5. Single dataset, single (binary) sensitive attribute.** Everything here is "
        "Adult Income / gender. That's a fine proof-of-concept, but a claim like "
        "'Adaptive Alpha Control improves fairness–accuracy tradeoffs' generalizes only "
        "as far as you've tested it — one dataset and one binary sensitive attribute is "
        "one data point on generalizability, not a general result yet.\n\n"
        "**6. The accuracy floor partly manufactures the 'accuracy preserved' result.** "
        "The controller overrides fairness pressure whenever accuracy dips below 0.84 — "
        "so 'Dynamic keeps accuracy comparable to baseline' is, to some extent, an "
        "engineered constraint rather than a purely emergent finding. That's a "
        "legitimate design choice, but it should be disclosed as such rather than "
        "presented as a surprising discovery.\n\n"
        "**7. Held-out evaluation.** Worth double-checking (not verifiable from this "
        "CSV alone) that ACC/DEO/DAO are computed on data the controller's own "
        "pressure signal never saw — otherwise the controller could be adapting to "
        "the same data it's being scored on."
    )

    st.divider()
    st.markdown("#### Live significance test (computed from your loaded data)")
    st.caption(
        "Compares Dynamic Alpha against every other loaded group on the metric you "
        "choose. Uses a paired test (Wilcoxon signed-rank, matched by seed) when every "
        "seed appears exactly once in both groups, otherwise an unpaired test "
        "(Mann-Whitney U). p-values are Holm-corrected for testing against multiple "
        "groups at once."
    )
    sig_metric = st.selectbox("Metric to test", ["DEO", "DAO", "ACC"], key="sig_metric")
    other_groups_available = [g for g in GROUP_ORDER if g != DYNAMIC_LABEL]

    if dynamic_row.empty or not other_groups_available:
        st.info("Need Dynamic Alpha plus at least one other group loaded to run tests.")
    else:
        sig_table = build_significance_table(df, sig_metric, DYNAMIC_LABEL, other_groups_available)
        if sig_table.empty:
            st.info("Not enough data to run comparisons.")
        else:
            fmt = {
                "Mean diff": "{:.4f}", "p-value (raw)": "{:.4g}",
                "Effect size": "{:.2f}",
            }
            if "p-value (Holm-corrected)" in sig_table.columns:
                fmt["p-value (Holm-corrected)"] = "{:.4g}"
            st.dataframe(sig_table.style.format(fmt), use_container_width=True, hide_index=True)
            st.caption(
                "'Mean diff' is Dynamic minus the comparison group (negative = Dynamic is "
                "lower). Effect size is Cohen's dz (paired) or Cohen's d (unpaired) — "
                "roughly, 0.2 = small, 0.5 = medium, 0.8+ = large. A statistically "
                "significant result with a tiny effect size is a real but practically "
                "negligible difference; treat both columns together, not p-values alone."
            )

# --------------------------------------------------------------------------
# Tab 6: Raw data
# --------------------------------------------------------------------------

with tab_data:
    st.subheader("Underlying run data")
    show_cols = [c for c in [
        "Name", "group", "seed_final", "alpha_used", "ACC", "DEO", "DAO",
        "pred_loss", "adv_loss", "pressure", "ema_dacc", "ema_ddao", "ema_ddeo",
    ] if c in df.columns]
    st.dataframe(df[show_cols].sort_values(["group", "seed_final"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download parsed data as CSV",
        df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="adaptive_alpha_control_parsed.csv",
        mime="text/csv",
    )