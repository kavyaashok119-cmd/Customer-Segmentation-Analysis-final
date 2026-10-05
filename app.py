"""
Customer Segmentation Analysis Dashboard
=========================================
A Streamlit dashboard that groups customers into segments using
K-Means clustering and explains each segment in plain English.

Run with:   streamlit run app.py
"""

from pathlib import Path
import io

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# 1. CONSTANTS
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"
ROOT_PATH = BASE_DIR / "customer_data.csv"

# Columns used by K-Means
FEATURES = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "AveragePurchaseValue",
    "TotalPurchases",
]
CATEGORICAL = ["Gender", "ProductCategory", "Location"]
REQUIRED_COLUMNS = ["CustomerID"] + FEATURES + CATEGORICAL

# Fixed colours for well-known segment names (any other name gets a fallback colour)
SEGMENT_COLORS = {
    "High-Value Customers": "#2563EB",
    "Premium Customers": "#7C3AED",
    "Potential Customers": "#F59E0B",
    "Budget Customers": "#10B981",
    "Low-Engagement Customers": "#EF4444",
}
FALLBACK_COLORS = ["#6B7280", "#EC4899", "#14B8A6", "#A16207", "#0EA5E9", "#84CC16"]

PAGES = [
    "📊 Overview",
    "🧩 Customer Segmentation",
    "🛍️ Customer Behavior",
    "📈 Segment Analysis",
    "💡 Business Insights",
    "🗂️ Data Explorer",
]

sns.set_theme(style="whitegrid", font_scale=0.9)


# ----------------------------------------------------------------------
# 2. SAMPLE DATA GENERATOR (used only if no CSV file exists)
# ----------------------------------------------------------------------
def generate_sample_data(n_customers=600, seed=42):
    """Create a realistic customer dataset with five hidden customer types."""
    rng = np.random.default_rng(seed)

    # (share, age, income, spending, frequency, avg purchase value) as (mean, std)
    groups = [
        (0.18, (38, 8), (90000, 10000), (82, 8), (40, 6), (120, 20)),   # high value
        (0.15, (45, 8), (105000, 12000), (55, 10), (12, 3), (350, 50)),  # premium
        (0.27, (35, 9), (55000, 8000), (52, 8), (22, 5), (70, 15)),      # potential
        (0.25, (28, 7), (28000, 6000), (25, 8), (14, 4), (30, 8)),       # budget
        (0.15, (50, 9), (60000, 10000), (30, 8), (4, 2), (45, 12)),      # low engagement
    ]
    counts = [int(round(g[0] * n_customers)) for g in groups]
    counts[-1] = n_customers - sum(counts[:-1])

    rows = []
    for g, c in zip(groups, counts):
        block = {
            "Age": rng.normal(*g[1], c),
            "AnnualIncome": rng.normal(*g[2], c),
            "SpendingScore": rng.normal(*g[3], c),
            "PurchaseFrequency": rng.normal(*g[4], c),
            "AveragePurchaseValue": rng.normal(*g[5], c),
        }
        rows.append(pd.DataFrame(block))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Keep values in realistic ranges
    df["Age"] = df["Age"].clip(18, 70).round().astype(int)
    df["AnnualIncome"] = (df["AnnualIncome"].clip(12000, 160000) / 100).round() * 100
    df["SpendingScore"] = df["SpendingScore"].clip(1, 100).round().astype(int)
    df["PurchaseFrequency"] = df["PurchaseFrequency"].clip(1, 60).round().astype(int)
    df["AveragePurchaseValue"] = df["AveragePurchaseValue"].clip(10, 500).round(2)

    # TotalPurchases = purchases per year x years as a customer (1-5)
    tenure = rng.uniform(1, 5, n_customers)
    df["TotalPurchases"] = np.maximum(1, (df["PurchaseFrequency"] * tenure).round()).astype(int)

    df.insert(0, "CustomerID", [f"CUST{i:04d}" for i in range(1, n_customers + 1)])
    df.insert(1, "Gender", rng.choice(["Male", "Female"], n_customers, p=[0.47, 0.53]))
    df["ProductCategory"] = rng.choice(
        ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Groceries", "Books"],
        n_customers,
        p=[0.18, 0.20, 0.15, 0.12, 0.10, 0.15, 0.10],
    )
    df["Location"] = rng.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle", "Miami"],
        n_customers,
    )

    # Add a few missing values and duplicate rows so the cleaning step has work to do
    for col, how_many in [("Age", 4), ("AnnualIncome", 3), ("Location", 3)]:
        idx = rng.choice(df.index, how_many, replace=False)
        df[col] = df[col].astype(object) if col == "Location" else df[col].astype(float)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(6, random_state=seed)], ignore_index=True)
    return df


# ----------------------------------------------------------------------
# 3. DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes=None):
    """Load the dataset from an uploaded file, or from the bundled CSV."""
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if ROOT_PATH.exists():
        return pd.read_csv(ROOT_PATH)
    return generate_sample_data()  # last resort


@st.cache_data(show_spinner=False)
def preprocess(raw):
    """Clean the data. Returns (clean_dataframe, report_dictionary)."""
    df = raw.copy()
    report = {
        "rows_before": len(df),
        "missing_before": raw.isna().sum(),
    }

    # Step 1: remove duplicate rows (exact copies, then repeated CustomerIDs)
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="CustomerID", keep="first")
    report["duplicates_removed"] = before - len(df)

    # Step 2: make sure numeric columns really are numbers
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 3: fill missing values (median for numbers, most common value for text)
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")

    # Step 4: convert categorical text into numeric codes (for reference / ML use)
    for col in CATEGORICAL:
        df[col + "_Code"] = df[col].astype("category").cat.codes

    # Step 5: helper column - estimated revenue per customer
    df["Revenue"] = df["TotalPurchases"] * df["AveragePurchaseValue"]

    report["rows_after"] = len(df)
    report["missing_after"] = df[REQUIRED_COLUMNS].isna().sum()
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------
# 4. K-MEANS CLUSTERING
# ----------------------------------------------------------------------
def scale_features(df):
    """Standardise features: mean = 0, standard deviation = 1."""
    return StandardScaler().fit_transform(df[FEATURES])


@st.cache_data(show_spinner=False)
def compute_elbow(df, k_max=10):
    """Run K-Means for k = 1..k_max and pick the 'elbow' of the inertia curve."""
    X = scale_features(df)
    ks = list(range(1, k_max + 1))
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]

    # Elbow = the point furthest from the straight line joining the first and last points
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))
    distance = np.abs(x + y - 1)
    optimal_k = ks[int(np.argmax(distance))]
    return ks, inertias, optimal_k


def name_segments(df):
    """
    Give every cluster a meaningful name based on its real statistics.

    1. Compare each cluster's average to the overall average (as a z-score).
    2. Score every cluster against each segment type.
    3. Give each segment name to the best-matching cluster that is still unnamed.
    """
    means = df.groupby("Cluster")[FEATURES].mean()
    z = (means - df[FEATURES].mean()) / df[FEATURES].std()

    # (name, scoring function, minimum score needed to earn the name)
    segment_types = [
        ("High-Value Customers", lambda r: r.AnnualIncome + r.SpendingScore + r.PurchaseFrequency, 0.5),
        ("Premium Customers", lambda r: r.AnnualIncome + r.AveragePurchaseValue, 0.5),
        ("Budget Customers", lambda r: -(r.AnnualIncome + r.SpendingScore), 0.5),
        ("Potential Customers", lambda r: -(abs(r.AnnualIncome) + abs(r.SpendingScore)), -99),
        ("Low-Engagement Customers", lambda r: -(r.PurchaseFrequency + r.TotalPurchases), 0.3),
    ]

    names, free = {}, list(z.index)
    for name, score_fn, min_score in segment_types:
        if not free:
            break
        scores = {c: score_fn(z.loc[c]) for c in free}
        best = max(scores, key=scores.get)
        if scores[best] >= min_score:
            names[best] = name
            free.remove(best)

    # Any cluster left over gets a generic name
    for i, c in enumerate(free, start=1):
        names[c] = "Regular Customers" if len(free) == 1 else f"Regular Customers {i}"
    return names


@st.cache_data(show_spinner=False)
def segment_customers(df, k):
    """Run K-Means with k clusters. Returns (clustered_df, silhouette, pca_variance)."""
    X = scale_features(df)
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)

    out = df.copy()
    out["Cluster"] = model.labels_
    out["Segment"] = out["Cluster"].map(name_segments(out))

    # 2-D projection (PCA) just for drawing the clusters
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    out["PC1"], out["PC2"] = coords[:, 0], coords[:, 1]

    sil = silhouette_score(X, model.labels_) if k > 1 else float("nan")
    return out, sil, pca.explained_variance_ratio_.sum()


# ----------------------------------------------------------------------
# 5. SUMMARIES, INSIGHTS AND RECOMMENDATIONS (all generated from the data)
# ----------------------------------------------------------------------
def color_map(segments):
    """Map each segment name to a colour."""
    cmap, spare = {}, iter(FALLBACK_COLORS)
    for s in sorted(segments):
        cmap[s] = SEGMENT_COLORS.get(s) or next(spare, "#6B7280")
    return cmap


def segment_summary(df):
    """One row per segment with the key averages."""
    s = df.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        AvgAge=("Age", "mean"),
        AvgIncome=("AnnualIncome", "mean"),
        AvgSpending=("SpendingScore", "mean"),
        AvgPurchaseValue=("AveragePurchaseValue", "mean"),
        PurchaseFrequency=("PurchaseFrequency", "mean"),
        TotalRevenue=("Revenue", "sum"),
    )
    return s.sort_values("Customers", ascending=False).reset_index()


def level(value, series):
    """Turn a number into Low / Medium / High compared with all customers."""
    z = (value - series.mean()) / series.std() if series.std() > 0 else 0
    return "High" if z > 0.5 else "Low" if z < -0.5 else "Medium"


STRATEGIES = {
    "High-Value Customers": (
        "Target them with premium products, loyalty rewards, exclusive offers, "
        "early access to new launches and personalised recommendations."
    ),
    "Premium Customers": (
        "Promote high-end and limited-edition products, offer concierge-style service, "
        "bundles and premium memberships. Encourage more frequent purchases with "
        "'complete the look' style suggestions."
    ),
    "Potential Customers": (
        "Nudge them upward with personalised offers, cross-selling, product bundles and "
        "a points-based loyalty scheme. They have room to grow into high-value customers."
    ),
    "Budget Customers": (
        "Use discounts, seasonal sales, coupons, value packs and buy-more-save-more deals. "
        "Keep communication simple and focus on affordability."
    ),
    "Low-Engagement Customers": (
        "Run re-engagement campaigns: win-back emails, 'we miss you' discounts, reminders "
        "and surveys to learn why they buy so rarely."
    ),
}


def describe_segment(name, seg, everyone):
    """Build the plain-English insight text for one segment."""
    n = len(seg)
    inc, spend = seg["AnnualIncome"].mean(), seg["SpendingScore"].mean()
    freq, apv = seg["PurchaseFrequency"].mean(), seg["AveragePurchaseValue"].mean()
    inc_l = level(inc, everyone["AnnualIncome"])
    spend_l = level(spend, everyone["SpendingScore"])
    freq_l = level(freq, everyone["PurchaseFrequency"])
    apv_l = level(apv, everyone["AveragePurchaseValue"])

    top_gender = seg["Gender"].mode().iloc[0]
    gender_pct = (seg["Gender"] == top_gender).mean() * 100

    # Use the preset strategy if the name is known, otherwise build one from the levels
    if name in STRATEGIES:
        strategy = STRATEGIES[name]
    elif spend_l == "High":
        strategy = "Reward them with loyalty points and personalised offers to keep spending high."
    elif freq_l == "Low":
        strategy = "Re-engage them with reminders and time-limited offers to raise purchase frequency."
    else:
        strategy = "Use targeted promotions and cross-selling to move them toward higher spending."

    return {
        "who": (
            f"{n:,} customers with an average age of {seg['Age'].mean():.0f}. "
            f"{gender_pct:.0f}% are {top_gender.lower()}. Most popular category: "
            f"{seg['ProductCategory'].mode().iloc[0]}; most common location: "
            f"{seg['Location'].mode().iloc[0]}."
        ),
        "behavior": (
            f"They make about {freq:.0f} purchases a year with an average purchase of "
            f"${apv:,.0f} ({apv_l.lower()} basket size), generating ${seg['Revenue'].sum():,.0f} in total revenue."
        ),
        "income": f"{inc_l} - average annual income of ${inc:,.0f}.",
        "spending": f"{spend_l} - average spending score of {spend:.0f} out of 100.",
        "frequency": f"{freq_l} - about {freq:.1f} purchases per year.",
        "strategy": strategy,
    }


def build_recommendations(df, full):
    """Create business recommendations using the numbers in the (filtered) data."""
    recs = []
    total_rev = df["Revenue"].sum()
    present = set(df["Segment"].unique())

    def seg(name):
        return df[df["Segment"] == name]

    if "High-Value Customers" in present:
        s = seg("High-Value Customers")
        share_c, share_r = len(s) / len(df) * 100, s["Revenue"].sum() / total_rev * 100
        recs.append((
            "🏆 Launch a loyalty programme for High-Value Customers",
            f"They are only {share_c:.0f}% of customers but bring in {share_r:.0f}% of revenue. "
            f"Protect them with reward points, VIP offers and early access. "
            f"Their favourite category is {s['ProductCategory'].mode().iloc[0]}.",
        ))
    if "Budget Customers" in present:
        s = seg("Budget Customers")
        recs.append((
            "🏷️ Offer discounts to Budget Customers",
            f"{len(s):,} customers earn about ${s['AnnualIncome'].mean():,.0f} a year and spend "
            f"${s['AveragePurchaseValue'].mean():,.0f} per purchase. Coupons, value packs and "
            f"seasonal sales will suit them best.",
        ))
    if "Potential Customers" in present:
        s = seg("Potential Customers")
        recs.append((
            "🎯 Send personalised offers to Potential Customers",
            f"{len(s):,} mid-range customers could grow. Recommend products from "
            f"{s['ProductCategory'].mode().iloc[0]} and bundle items to raise their "
            f"${s['AveragePurchaseValue'].mean():,.0f} average purchase.",
        ))
    if "Premium Customers" in present:
        s = seg("Premium Customers")
        recs.append((
            "💎 Promote premium products to high-income customers",
            f"Premium Customers spend ${s['AveragePurchaseValue'].mean():,.0f} per purchase but buy only "
            f"{s['PurchaseFrequency'].mean():.0f} times a year. Showcase high-end items and "
            f"subscriptions to increase how often they buy.",
        ))

    # Low-frequency customers across all segments (bottom 25% of purchase frequency)
    cutoff = full["PurchaseFrequency"].quantile(0.25)
    low = df[df["PurchaseFrequency"] <= cutoff]
    if len(low):
        recs.append((
            "🔔 Run re-engagement campaigns for low-frequency customers",
            f"{len(low):,} customers ({len(low) / len(df) * 100:.0f}%) buy {cutoff:.0f} times a year or "
            f"fewer. Win-back emails and reminders could bring them back, especially in "
            f"{low['Location'].mode().iloc[0]}.",
        ))

    # Best category and location by revenue
    cat = df.groupby("ProductCategory")["Revenue"].sum().idxmax()
    loc = df.groupby("Location")["Revenue"].sum().idxmax()
    recs.append((
        "📦 Focus marketing where revenue is strongest",
        f"{cat} is the top revenue category and {loc} is the top revenue location. "
        f"Prioritise stock, ads and promotions there.",
    ))
    return recs


# ----------------------------------------------------------------------
# 6. CHART HELPERS (Matplotlib + Seaborn)
# ----------------------------------------------------------------------
def show(fig):
    """Display a Matplotlib figure in Streamlit and free its memory."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def chart_segment_counts(df, cmap):
    counts = df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Customers"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=counts, x="Segment", y="Customers", hue="Segment", palette=cmap, legend=False, ax=ax)
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height())}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Customers per Segment")
    show(fig)


def chart_scatter(df, x, y, cmap, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, hue="Segment", palette=cmap, s=40, alpha=0.8, ax=ax)
    ax.set_title(title)
    ax.legend(fontsize=7, title=None)
    show(fig)


def chart_hist(df, col, title, color="#2563EB"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col], bins=25, kde=True, color=color, ax=ax)
    ax.set_title(title)
    show(fig)


def chart_mean_by_segment(df, col, title, cmap):
    means = df.groupby("Segment")[col].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=means, x="Segment", y=col, hue="Segment", palette=cmap, legend=False, ax=ax)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():,.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(title)
    show(fig)


def chart_counts(df, col, title, palette="Blues_d"):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Customers"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=counts, x=col, y="Customers", hue=col, palette=palette, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title(title)
    show(fig)


def chart_clusters(df, cmap, variance):
    """2-D cluster picture using PCA (6 features squeezed into 2 axes)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Segment", palette=cmap, s=45, alpha=0.85, ax=ax)
    centers = df.groupby("Segment")[["PC1", "PC2"]].mean()
    ax.scatter(centers["PC1"], centers["PC2"], marker="X", s=180, c="black", label="Centre")
    ax.set_title(f"Customer Clusters (PCA view, {variance * 100:.0f}% of variation shown)")
    ax.legend(fontsize=7, title=None)
    show(fig)


def chart_elbow(ks, inertias, optimal_k):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o", color="#2563EB")
    ax.axvline(optimal_k, color="#EF4444", linestyle="--", label=f"Optimal k = {optimal_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster distance)")
    ax.set_title("Elbow Method")
    ax.set_xticks(ks)
    ax.legend()
    show(fig)


def chart_heatmap(df):
    """How each segment compares with the average customer (z-scores)."""
    cols = FEATURES
    seg_means = df.groupby("Segment")[cols].mean()
    z = (seg_means - df[cols].mean()) / df[cols].std()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(z, annot=True, fmt=".1f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Segment profile vs. average customer (green = above, red = below)")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    show(fig)


# ----------------------------------------------------------------------
# 7. STYLING AND SMALL UI HELPERS
# ----------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .main-title {background: linear-gradient(90deg,#1E3A8A,#2563EB); color:white;
                     padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 1rem;}
        .main-title h1 {color:white; font-size:1.7rem; margin:0; padding:0;}
        .main-title p {color:#DBEAFE; margin:0.2rem 0 0 0; font-size:0.95rem;}
        .kpi {background:#FFFFFF; border:1px solid #E5E7EB; border-left:5px solid #2563EB;
              border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.6rem;
              box-shadow: 0 1px 3px rgba(0,0,0,0.06);}
        .kpi-label {color:#6B7280; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em;}
        .kpi-value {color:#111827; font-size:1.45rem; font-weight:700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title, subtitle):
    st.markdown(f'<div class="main-title"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(df):
    """Six KPI cards."""
    items = [
        ("Total Customers", f"{len(df):,}"),
        ("Average Income", f"${df['AnnualIncome'].mean():,.0f}"),
        ("Avg Spending Score", f"{df['SpendingScore'].mean():.1f}"),
        ("Avg Purchase Value", f"${df['AveragePurchaseValue'].mean():,.0f}"),
        ("Total Revenue", f"${df['Revenue'].sum():,.0f}"),
        ("Segments", f"{df['Segment'].nunique()}"),
    ]
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def range_slider(label, series):
    """A min-max slider for a numeric column (skipped if all values are equal)."""
    low, high = int(np.floor(series.min())), int(np.ceil(series.max()))
    if low == high:
        return low, high
    step = 1000 if high - low > 5000 else 1
    return st.sidebar.slider(label, low, high, (low, high), step=step)


def sidebar_filters(df):
    """Draw the filter widgets and return the filtered dataframe."""
    st.sidebar.markdown("### 🔎 Filters")
    genders = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    age = range_slider("Age", df["Age"])
    income = range_slider("Annual income ($)", df["AnnualIncome"])
    spending = range_slider("Spending score", df["SpendingScore"])
    segments = st.sidebar.multiselect("Customer segment", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))
    cats = st.sidebar.multiselect("Product category", sorted(df["ProductCategory"].unique()), default=sorted(df["ProductCategory"].unique()))
    locs = st.sidebar.multiselect("Location", sorted(df["Location"].unique()), default=sorted(df["Location"].unique()))

    mask = (
        df["Gender"].isin(genders)
        & df["Age"].between(*age)
        & df["AnnualIncome"].between(*income)
        & df["SpendingScore"].between(*spending)
        & df["Segment"].isin(segments)
        & df["ProductCategory"].isin(cats)
        & df["Location"].isin(locs)
    )
    return df[mask]


# ----------------------------------------------------------------------
# 8. DASHBOARD PAGES
# ----------------------------------------------------------------------
def page_overview(df, cmap):
    header("Customer Segmentation Dashboard", "Understand your customers and target them with the right strategy")
    kpi_row(df)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Customer Segment Distribution")
        chart_segment_counts(df, cmap)
        st.caption("Taller bars = more customers in that segment.")
    with c2:
        st.subheader("Annual Income Distribution")
        chart_hist(df, "AnnualIncome", "Annual Income Distribution")
        st.caption("Shows how many customers fall into each income band.")
    st.subheader("Segment Summary")
    show_summary_table(df)


def show_summary_table(df):
    summary = segment_summary(df)
    table = summary.rename(columns={
        "Customers": "Customer Count", "AvgAge": "Average Age", "AvgIncome": "Average Income",
        "AvgSpending": "Average Spending Score", "AvgPurchaseValue": "Average Purchase Value",
        "PurchaseFrequency": "Purchase Frequency", "TotalRevenue": "Total Revenue",
    })
    st.dataframe(
        table.style.format({
            "Average Age": "{:.1f}", "Average Income": "${:,.0f}", "Average Spending Score": "{:.1f}",
            "Average Purchase Value": "${:,.2f}", "Purchase Frequency": "{:.1f}", "Total Revenue": "${:,.0f}",
        }),
        hide_index=True,
    )


def page_segmentation(df, cmap, ctx):
    header("Customer Segmentation", "K-Means clustering on six customer features")
    ks, inertias, optimal_k = ctx["elbow"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimal clusters (Elbow)", optimal_k)
    m2.metric("Clusters used", ctx["k"])
    m3.metric("Silhouette score", f"{ctx['silhouette']:.2f}")
    st.caption("Silhouette score ranges from -1 to 1. Higher means the clusters are better separated.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Elbow Curve")
        chart_elbow(ks, inertias, optimal_k)
        st.caption("Pick the k where the curve bends - adding more clusters after that gives little benefit.")
    with c2:
        st.subheader("Cluster Visualization")
        chart_clusters(df, cmap, ctx["variance"])
        st.caption("Each dot is a customer; the black X marks the centre of each segment.")

    st.subheader("Cluster Labels for Each Customer")
    labels = df[["CustomerID", "Cluster", "Segment"]]
    st.dataframe(labels, hide_index=True)
    st.subheader("Cluster Number to Segment Name")
    st.dataframe(df.groupby(["Cluster", "Segment"]).size().reset_index(name="Customers"), hide_index=True)


def page_behavior(df, cmap):
    header("Customer Behavior", "How customers earn, spend and shop")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Income vs Spending Score")
        chart_scatter(df, "AnnualIncome", "SpendingScore", cmap, "Income vs Spending Score")
        st.caption("Each colour is a segment. Top-right = high income and high spending.")
    with c2:
        st.subheader("Age vs Spending Score")
        chart_scatter(df, "Age", "SpendingScore", cmap, "Age vs Spending Score")
        st.caption("Shows which age groups spend the most.")

    st.subheader("Customer Demographics")
    d1, d2 = st.columns(2)
    with d1:
        chart_counts(df, "Gender", "Gender Distribution", palette="Set2")
    with d2:
        chart_hist(df, "Age", "Age Distribution", color="#7C3AED")

    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Product Categories")
        chart_counts(df, "ProductCategory", "Customers by Product Category")
    with e2:
        st.subheader("Locations")
        chart_counts(df, "Location", "Customers by Location", palette="Greens_d")


def page_segment_analysis(df, cmap):
    header("Segment Analysis", "Compare segments side by side")
    st.subheader("Segment Summary")
    show_summary_table(df)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Purchase Frequency by Segment")
        chart_mean_by_segment(df, "PurchaseFrequency", "Average Purchases per Year", cmap)
        st.caption("How often each segment buys.")
    with c2:
        st.subheader("Average Purchase Value by Segment")
        chart_mean_by_segment(df, "AveragePurchaseValue", "Average Purchase Value ($)", cmap)
        st.caption("How much each segment spends per purchase.")

    st.subheader("Segment Profile Heatmap")
    chart_heatmap(df)
    st.caption("Green cells mean the segment is above average on that feature; red means below average.")


def page_insights(df, full, cmap):
    header("Business Insights", "Automatically generated from the cluster statistics")
    st.subheader("Customer Insights by Segment")
    for name in segment_summary(df)["Segment"]:
        info = describe_segment(name, df[df["Segment"] == name], full)
        with st.container(border=True):
            st.markdown(f"#### {name}")
            st.markdown(f"**Who they are:** {info['who']}")
            st.markdown(f"**Purchasing behavior:** {info['behavior']}")
            st.markdown(f"**Income level:** {info['income']}")
            st.markdown(f"**Spending behavior:** {info['spending']}")
            st.markdown(f"**Purchase frequency:** {info['frequency']}")
            st.success(f"**Recommended marketing strategy:** {info['strategy']}")

    st.subheader("Business Recommendations")
    for title, text in build_recommendations(df, full):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)


def page_explorer(df, full, raw, report, original_cols):
    header("Data Explorer", "Inspect, clean and download the data")
    tab_raw, tab_quality, tab_result = st.tabs(["Raw Dataset", "Data Quality", "Segmented Data"])

    with tab_raw:
        st.write(f"Raw file: **{len(raw):,} rows** x **{raw.shape[1]} columns**")
        st.dataframe(raw)

    with tab_quality:
        q1, q2, q3 = st.columns(3)
        q1.metric("Rows before cleaning", f"{report['rows_before']:,}")
        q2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        q3.metric("Rows after cleaning", f"{report['rows_after']:,}")
        missing = pd.DataFrame({
            "Missing before": report["missing_before"],
            "Missing after": report["missing_after"],
        }).fillna(0).astype(int)
        st.write("**Missing values per column** (numbers filled with the median, text with the most common value):")
        st.dataframe(missing)

    with tab_result:
        st.write(f"Showing **{len(df):,}** of {len(full):,} customers (sidebar filters applied).")
        export_cols = original_cols + ["Cluster", "Segment"]
        st.dataframe(df[export_cols], hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download all segmented customers (CSV)",
            full[export_cols].to_csv(index=False).encode("utf-8"),
            "customer_segments.csv", "text/csv",
        )
        d2.download_button(
            "⬇️ Download filtered customers (CSV)",
            df[export_cols].to_csv(index=False).encode("utf-8"),
            "customer_segments_filtered.csv", "text/csv",
        )


# ----------------------------------------------------------------------
# 9. MAIN APP
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")
    inject_css()

    # ---- Sidebar: navigation, data source, clustering settings ----
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

    st.sidebar.markdown("### 📁 Data")
    uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")

    raw = load_data(uploaded.getvalue() if uploaded is not None else None)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        st.error(f"Your CSV is missing these required columns: {', '.join(missing_cols)}")
        st.stop()

    # ---- Preprocess and find the optimal number of clusters ----
    clean, report = preprocess(raw)
    ks, inertias, optimal_k = compute_elbow(clean)

    st.sidebar.markdown("### ⚙️ Clustering")
    use_auto = st.sidebar.checkbox("Use Elbow Method result", value=True)
    k = optimal_k if use_auto else st.sidebar.slider("Number of clusters", 2, 8, optimal_k)

    full, silhouette, variance = segment_customers(clean, k)
    cmap = color_map(full["Segment"].unique())
    ctx = {"elbow": (ks, inertias, optimal_k), "k": k, "silhouette": silhouette, "variance": variance}

    # ---- Filters ----
    df = sidebar_filters(full)
    if df.empty:
        st.warning("No customers match the current filters. Please widen your selection in the sidebar.")
        st.stop()

    # ---- Pages ----
    if page == PAGES[0]:
        page_overview(df, cmap)
    elif page == PAGES[1]:
        page_segmentation(df, cmap, ctx)
    elif page == PAGES[2]:
        page_behavior(df, cmap)
    elif page == PAGES[3]:
        page_segment_analysis(df, cmap)
    elif page == PAGES[4]:
        page_insights(df, full, cmap)
    else:
        page_explorer(df, full, raw, report, list(raw.columns))


if __name__ == "__main__":
    main()
