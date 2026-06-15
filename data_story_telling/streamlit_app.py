import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Store Sales & Profit Storytelling Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "synthetic_store_sales.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "sales", "profit"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.month_name()
    df["quarter"] = "Q" + df["order_date"].dt.quarter.astype(str)
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["profit_margin"] = np.where(df["sales"] != 0, df["profit"] / df["sales"] * 100, 0)
    df["average_selling_price"] = np.where(df["quantity"] != 0, df["sales"] / df["quantity"], 0)
    df["discount_range"] = pd.cut(
        df["discount"],
        bins=[-0.01, 0, 0.10, 0.20, 0.30, 0.50, 1.00],
        labels=["No Discount", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"],
    )
    return df

def money(value: float) -> str:
    return f"${value:,.0f}"

def pct(value: float) -> str:
    return f"{value:,.2f}%"

def grouped_performance(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.groupby(group_col, observed=False).agg(
        sales=("sales", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("order_id", "nunique"),
        avg_discount=("discount", "mean"),
        avg_shipping_cost=("shipping_cost", "mean"),
    ).reset_index()
    out["profit_margin"] = np.where(out["sales"] != 0, out["profit"] / out["sales"] * 100, 0)
    return out

def recommendation_engine(df: pd.DataFrame) -> list[str]:
    recs = []
    total_sales = df["sales"].sum()
    total_profit = df["profit"].sum()
    margin = total_profit / total_sales * 100 if total_sales else 0

    if margin < 10:
        recs.append("Overall profit margin is below 10%. Review pricing, discounts, and product cost structure.")
    else:
        recs.append("Overall margin is healthy, but category-level leakage should still be monitored.")

    high_discount_loss = df[(df["discount"] > 0.20) & (df["profit"] < 0)]
    if len(high_discount_loss):
        recs.append("Transactions with discounts above 20% frequently create losses. Add approval rules for high discounts.")

    subcat = grouped_performance(df, "sub_category")
    loss_subcat = subcat[subcat["profit"] < 0].sort_values("profit").head(3)
    for _, row in loss_subcat.iterrows():
        recs.append(f"Investigate {row['sub_category']} because it generated {money(row['profit'])} profit.")

    region = grouped_performance(df, "region")
    weak_region = region.sort_values("profit_margin").head(1)
    if not weak_region.empty:
        row = weak_region.iloc[0]
        recs.append(f"Review regional pricing in {row['region']}; margin is only {pct(row['profit_margin'])}.")

    product = grouped_performance(df, "product_name")
    worst_product = product.sort_values("profit").head(1)
    if not worst_product.empty:
        row = worst_product.iloc[0]
        recs.append(f"Reprice or bundle '{row['product_name']}' because it is the weakest product by profit.")

    return recs[:7]

try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("CSV file not found. Put synthetic_store_sales.csv in the same folder as streamlit_app.py.")
    st.stop()

st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Use filters to explore business performance and profit leakage.")

min_date = df_raw["order_date"].min().date()
max_date = df_raw["order_date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

regions = st.sidebar.multiselect("Region", sorted(df_raw["region"].dropna().unique()), default=sorted(df_raw["region"].dropna().unique()))
categories = st.sidebar.multiselect("Category", sorted(df_raw["category"].dropna().unique()), default=sorted(df_raw["category"].dropna().unique()))
segments = st.sidebar.multiselect("Customer segment", sorted(df_raw["segment"].dropna().unique()), default=sorted(df_raw["segment"].dropna().unique()))

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

filtered = df_raw[
    (df_raw["order_date"] >= start_date)
    & (df_raw["order_date"] <= end_date)
    & (df_raw["region"].isin(regions))
    & (df_raw["category"].isin(categories))
    & (df_raw["segment"].isin(segments))
].copy()

st.title("📊 Store Sales & Profit Storytelling Dashboard")
st.markdown(
    "This dashboard turns transaction-level sales data into a business story: **what happened, why it happened, and what management should do next.**"
)

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

sales = filtered["sales"].sum()
profit = filtered["profit"].sum()
orders = filtered["order_id"].nunique()
customers = filtered["customer_name"].nunique()
margin = profit / sales * 100 if sales else 0
aov = sales / orders if orders else 0

best_category = grouped_performance(filtered, "category").sort_values("profit", ascending=False).iloc[0]["category"]
weak_category = grouped_performance(filtered, "category").sort_values("profit").iloc[0]["category"]
best_region = grouped_performance(filtered, "region").sort_values("profit", ascending=False).iloc[0]["region"]

st.subheader("1. Executive KPI Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Sales", money(sales))
col2.metric("Total Profit", money(profit))
col3.metric("Profit Margin", pct(margin))
col4.metric("Orders", f"{orders:,}")
col5.metric("Avg Order Value", money(aov))

with st.expander("Executive summary", expanded=True):
    st.write(
        f"The business generated **{money(sales)}** in sales and **{money(profit)}** in profit, with an overall margin of **{pct(margin)}**. "
        f"The strongest category is **{best_category}**, while **{weak_category}** needs attention. "
        f"The best region by profit is **{best_region}**. The main management focus should be profit leakage, high-discount transactions, and low-margin products."
    )

st.subheader("2. Data Quality Check")
missing = filtered.isna().sum().reset_index()
missing.columns = ["column", "missing_values"]
missing["missing_percentage"] = missing["missing_values"] / len(filtered) * 100
q1, q2, q3 = st.columns(3)
q1.metric("Rows", f"{len(filtered):,}")
q2.metric("Columns", f"{filtered.shape[1]:,}")
q3.metric("Duplicate Rows", f"{filtered.duplicated().sum():,}")
st.dataframe(missing[missing["missing_values"] > 0].sort_values("missing_values", ascending=False), use_container_width=True)

st.subheader("3. Revenue and Profit Trend")
monthly = filtered.groupby("year_month", observed=False).agg(sales=("sales", "sum"), profit=("profit", "sum")).reset_index()
monthly["profit_margin"] = np.where(monthly["sales"] != 0, monthly["profit"] / monthly["sales"] * 100, 0)
fig_sales = px.line(monthly, x="year_month", y="sales", markers=True, title="Monthly Sales Trend")
st.plotly_chart(fig_sales, use_container_width=True)
fig_profit = px.line(monthly, x="year_month", y="profit", markers=True, title="Monthly Profit Trend")
st.plotly_chart(fig_profit, use_container_width=True)

st.subheader("4. Category and Sub-Category Performance")
cat = grouped_performance(filtered, "category").sort_values("profit", ascending=False)
left, right = st.columns(2)
left.plotly_chart(px.bar(cat, x="category", y="sales", title="Sales by Category", text_auto=".2s"), use_container_width=True)
right.plotly_chart(px.bar(cat, x="category", y="profit", title="Profit by Category", text_auto=".2s"), use_container_width=True)
st.dataframe(cat.sort_values("profit", ascending=False), use_container_width=True)

subcat = grouped_performance(filtered, "sub_category")
left, right = st.columns(2)
left.plotly_chart(px.bar(subcat.sort_values("profit", ascending=False).head(10), x="profit", y="sub_category", orientation="h", title="Top 10 Profitable Sub-Categories"), use_container_width=True)
right.plotly_chart(px.bar(subcat.sort_values("profit").head(10), x="profit", y="sub_category", orientation="h", title="Top 10 Loss-Making Sub-Categories"), use_container_width=True)

st.subheader("5. Regional and Segment Performance")
region = grouped_performance(filtered, "region")
segment = grouped_performance(filtered, "segment")
left, right = st.columns(2)
left.plotly_chart(px.bar(region.sort_values("profit", ascending=False), x="region", y="profit", title="Profit by Region", text_auto=".2s"), use_container_width=True)
right.plotly_chart(px.bar(segment.sort_values("profit", ascending=False), x="segment", y="profit", title="Profit by Segment", text_auto=".2s"), use_container_width=True)

st.subheader("6. Product Intelligence")
product = grouped_performance(filtered, "product_name")
product["product_status"] = np.select(
    [
        (product["profit"] > 0) & (product["profit_margin"] >= 15),
        (product["profit"] > 0) & (product["profit_margin"] < 15),
        product["profit"] < 0,
    ],
    ["High Profit Product", "Low Margin Product", "Loss Making Product"],
    default="Neutral Product",
)
left, right = st.columns(2)
left.dataframe(product.sort_values("profit", ascending=False).head(15), use_container_width=True)
right.dataframe(product.sort_values("profit").head(15), use_container_width=True)
st.plotly_chart(px.pie(product, names="product_status", title="Product Risk Classification"), use_container_width=True)

st.subheader("7. Discount Impact Analysis")
disc = grouped_performance(filtered, "discount_range")
left, right = st.columns(2)
left.plotly_chart(px.scatter(filtered, x="discount", y="profit", color="category", hover_data=["product_name", "region", "sales"], title="Discount vs Profit"), use_container_width=True)
right.plotly_chart(px.bar(disc, x="discount_range", y="profit_margin", title="Profit Margin by Discount Range", text_auto=".2f"), use_container_width=True)
st.dataframe(disc, use_container_width=True)

st.subheader("8. Root Cause Analysis")
loss_data = filtered[filtered["profit"] < 0]
loss_summary = loss_data.groupby(["category", "sub_category"], observed=False).agg(
    loss_amount=("profit", "sum"),
    sales=("sales", "sum"),
    avg_discount=("discount", "mean"),
    avg_shipping_cost=("shipping_cost", "mean"),
    orders=("order_id", "nunique"),
).reset_index().sort_values("loss_amount")

c1, c2 = st.columns(2)
c1.metric("Loss-Making Orders", f"{len(loss_data):,}")
c2.metric("Total Loss Amount", money(loss_data["profit"].sum()))
st.dataframe(loss_summary.head(20), use_container_width=True)

heat = filtered.pivot_table(index="region", columns="category", values="profit", aggfunc="sum", observed=False).fillna(0)
st.plotly_chart(px.imshow(heat, text_auto=".2s", aspect="auto", title="Profit Heatmap: Region vs Category"), use_container_width=True)

st.subheader("9. Simple Sales Forecast")
forecast_periods = st.slider("Forecast months", min_value=3, max_value=12, value=6)
forecast_source = monthly.copy()
forecast_source["date"] = pd.to_datetime(forecast_source["year_month"])
forecast_source = forecast_source.sort_values("date")

if len(forecast_source) >= 3:
    recent = forecast_source.tail(min(6, len(forecast_source)))
    monthly_growth = recent["sales"].pct_change().replace([np.inf, -np.inf], np.nan).dropna().mean()
    if pd.isna(monthly_growth):
        monthly_growth = 0
    last_sales = forecast_source["sales"].iloc[-1]
    last_date = forecast_source["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=forecast_periods, freq="MS")
    forecast_values = [last_sales * ((1 + monthly_growth) ** i) for i in range(1, forecast_periods + 1)]
    forecast_df = pd.DataFrame({"date": future_dates, "sales": forecast_values, "type": "Forecast"})
    actual_df = forecast_source[["date", "sales"]].copy()
    actual_df["type"] = "Actual"
    forecast_plot = pd.concat([actual_df, forecast_df], ignore_index=True)
    st.plotly_chart(px.line(forecast_plot, x="date", y="sales", color="type", markers=True, title="Simple Sales Forecast"), use_container_width=True)
else:
    st.info("Not enough monthly data for forecasting.")

st.subheader("10. Strategic Recommendations")
for idx, rec in enumerate(recommendation_engine(filtered), start=1):
    st.markdown(f"**{idx}.** {rec}")

st.subheader("11. Download Filtered Analysis Data")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv, "filtered_store_sales_analysis.csv", "text/csv")

st.caption("Portfolio note: this dashboard is designed as a business storytelling layer over the Jupyter notebook analysis.")
