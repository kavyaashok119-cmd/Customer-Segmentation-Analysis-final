# 📊 Customer Segmentation Analysis Dashboard

## Project Objective
Build an interactive dashboard that analyses customer behaviour and demographics and
automatically groups customers into meaningful segments using **K-Means clustering**,
then explains each segment and recommends how to market to it.

## Problem Statement
A business with hundreds of customers cannot treat them all the same. Some customers spend a lot
and buy often, others are price-sensitive, and others have stopped buying. Without segmentation, marketing
money is wasted on the wrong people. This project uses unsupervised machine learning to discover
natural customer groups and turns the results into clear business actions.

## Technologies Used
| Tool | Purpose |
|------|---------|
| Python 3.9+ | Programming language |
| Pandas, NumPy | Data loading, cleaning and calculations |
| Scikit-learn | `StandardScaler`, `KMeans`, `PCA`, silhouette score |
| Matplotlib, Seaborn | Charts |
| Streamlit | Web dashboard |

## Project Structure
```
customer-segmentation/
├── app.py                 # The complete Streamlit application
├── customer_data.csv      # Sample dataset (606 rows incl. 6 duplicates)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── data/
    └── customer_data.csv  # Same sample dataset (default location the app reads)
```

## Dataset Description
The sample dataset has **600 unique customers** (606 rows - 6 duplicate rows and 10 missing values were
added on purpose so you can see the cleaning step work).

| Column | Description |
|--------|-------------|
| CustomerID | Unique customer ID |
| Gender | Male / Female |
| Age | Age in years |
| AnnualIncome | Yearly income in dollars (currency is just a label - change it freely) |
| SpendingScore | 1-100 score of how much the customer spends |
| PurchaseFrequency | Purchases per year |
| AveragePurchaseValue | Average amount per purchase |
| TotalPurchases | Total number of purchases made |
| ProductCategory | Most-bought category |
| Location | Customer city |

The app also creates **Revenue = TotalPurchases x AveragePurchaseValue** for revenue KPIs and charts.

## Data Preprocessing
1. **Load** the CSV (bundled sample, or one you upload).
2. **Display** the raw dataset (Data Explorer > Raw Dataset).
3. **Check missing values** (Data Explorer > Data Quality).
4. **Handle missing values** - median for numbers, most common value for text.
5. **Remove duplicates** - exact duplicate rows and repeated CustomerIDs.
6. **Convert categorical data** - Gender, ProductCategory and Location are also encoded as numeric `_Code` columns.
7. **Select features** - Age, AnnualIncome, SpendingScore, PurchaseFrequency, AveragePurchaseValue, TotalPurchases.
8. **Standardise** with `StandardScaler` so features with big numbers (income) do not dominate small ones (age).

## K-Means Clustering
K-Means groups customers so that people in the same cluster are as similar as possible. It works like this:
1. Choose the number of clusters *k*.
2. Place *k* centres randomly.
3. Assign every customer to the nearest centre.
4. Move each centre to the average of its customers.
5. Repeat steps 3-4 until nothing changes.

## Elbow Method
We run K-Means for k = 1 to 10 and record the **inertia** (total distance from customers to their cluster centre).
Inertia always falls as k grows, but after a certain point the improvement is tiny - the curve bends like an
elbow. The app finds that bend automatically (the point furthest from the straight line joining the first and last
points) and uses it as the default k. You can override it in the sidebar (2-8 clusters).

## Automatic Segment Naming
Names are **not hard-coded to cluster numbers**. After clustering, the app compares each cluster's averages with the
overall average (z-scores) and gives each name to the best-matching cluster:

| Segment | Matches the cluster with... |
|---------|------------------------------|
| High-Value Customers | highest income + spending + purchase frequency |
| Premium Customers | high income + highest average purchase value |
| Budget Customers | lowest income + spending |
| Potential Customers | income and spending closest to average |
| Low-Engagement Customers | lowest purchase frequency and total purchases |
| Regular Customers | any extra cluster that fits none of the above |

Because the names come from real statistics, they stay correct if you upload a different dataset or change k.

## Dashboard Features
* **Sidebar** - navigation, CSV upload, cluster settings and filters (gender, age, income, spending score, segment, product category, location).
* **6 KPI cards** - total customers, average income, average spending score, average purchase value, total revenue, number of segments.
* **Six sections** - Overview, Customer Segmentation, Customer Behavior, Segment Analysis, Business Insights, Data Explorer.
* **Charts** - segment distribution, income vs spending, age vs spending, income histogram, purchase frequency by segment,
  average purchase value by segment, gender and age distribution, 2-D cluster plot (PCA), elbow curve, segment heatmap.
* **Segment summary table**, **automatic insights**, **business recommendations** and **CSV download**.

## Customer Segments (sample data)
| Segment | Profile | Strategy |
|---------|---------|----------|
| High-Value | High income, high spending, very frequent buyers | Loyalty rewards, VIP offers, personalisation |
| Premium | Highest income, very large purchases, rare buying | Premium products, bundles, subscriptions |
| Potential | Mid income, mid spending | Personalised offers, cross-selling |
| Budget | Lower income, lower spending | Discounts, coupons, value packs |

## Business Insights
For each segment the dashboard writes who they are, their purchasing behaviour, income level, spending behaviour,
purchase frequency and a recommended marketing strategy. Recommendations (loyalty programme, discounts,
personalised offers, premium products, re-engagement campaigns) quote real numbers from your data.

## Installation
```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501.

## Use Your Own Dataset
Option A - **Upload in the app**: sidebar > *Upload your own CSV*.
Option B - **Replace the file**: overwrite `data/customer_data.csv`.

Your CSV must contain these columns (exact names):
`CustomerID, Gender, Age, AnnualIncome, SpendingScore, PurchaseFrequency, AveragePurchaseValue, TotalPurchases, ProductCategory, Location`.
Extra columns are kept in the download. If a column is missing, the app tells you which one.

## How to Read the Dashboard
* **Elbow curve** - the red dashed line is the suggested number of clusters.
* **Silhouette score** - closer to 1 means well-separated clusters (above 0.4 is decent for customer data).
* **Segment distribution** - how big each group is.
* **Income vs Spending scatter** - top-right dots are rich big spenders; bottom-left are budget customers.
* **Cluster plot (PCA)** - six features squeezed into two axes; separated colour blobs mean distinct segments.
* **Heatmap** - green = above average, red = below average for that feature.
* **Frequency / purchase value bars** - frequent small buyers vs rare big buyers.
* **Sidebar filters** change every chart, KPI, insight and recommendation. The clusters themselves are always calculated on all customers.

## Expected Output
* Elbow method suggests **4 clusters** on the sample data (silhouette about 0.41).
* Segments found: High-Value (~118), Premium (~90), Potential (~97), Budget (~295) customers.
* Six KPI cards, 10+ charts, a segment summary table, written insights, recommendations,
  and downloadable CSV files with `Cluster` and `Segment` columns.

## Troubleshooting
* *Command not found: streamlit* - run `python -m streamlit run app.py`.
* *Charts look empty* - your filters may be too narrow; widen them in the sidebar.
