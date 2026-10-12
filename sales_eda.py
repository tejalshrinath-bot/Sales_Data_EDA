# ============================================================
# SALES DATA EXPLORATORY DATA ANALYSIS (EDA)
# Industry: Retail
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. SETUP
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 60)
print("SALES DATA EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2. LOAD DATASET
file_path = Path("sales_data.csv")

# Create sample data if the CSV file does not exist.
# Replace this sample with your actual internship dataset.
if file_path.exists():
    df = pd.read_csv(file_path)
    print("Loaded sales_data.csv")
else:
    print("CSV not found. Creating sample dataset for practice.")

    sample_data = {
        "order_id": [1001, 1002, 1003, 1004, 1005,
                     1006, 1007, 1008, 1009, 1010,
                     1011, 1012],
        "date": [
            "2026-01-05", "2026-01-06", "2026-01-10",
            "2026-01-12", "2026-02-02", "2026-02-08",
            "2026-02-14", "2026-03-03", "2026-03-07",
            "2026-03-15", "2026-03-17", "2026-04-04"
        ],
        "product": [
            "Laptop", "Mouse", "Keyboard", "Laptop",
            "Monitor", "Mouse", "Keyboard", "Laptop",
            "Monitor", "Mouse", "Keyboard", "Monitor"
        ],
        "quantity": [2, 5, 3, 1, 2, 10, 4, 1, 3, 8, 2, 1],
        "price": [
            45000, 500, 1500, 45000, 12000, 500,
            1500, 45000, 12000, 500, 1500, 12000
        ],
        "region": [
            "West", "North", "East", "South",
            "West", "North", "East", "South",
            "West", "North", "East", "South"
        ]
    }

    df = pd.DataFrame(sample_data)

# 3. STANDARDIZE AND VALIDATE COLUMNS
df.columns = df.columns.str.strip().str.lower()

required_columns = [
    "order_id", "date", "product",
    "quantity", "price", "region"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df = df[required_columns].copy()

print("\nOriginal dataset preview:")
print(df.head())

print("\nOriginal dataset shape:", df.shape)

# 4. EXPLORE DATA
print("\nDataset information:")
df.info()

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 5. CLEAN DATA
initial_rows = len(df)

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Clean blank text values
for col in ["order_id", "product", "region"]:
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

# Convert dates and numeric columns
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Remove rows with missing essential values
df = df.dropna(subset=required_columns).copy()

# Exclude invalid quantities and prices
df = df[
    (df["quantity"] > 0) &
    (df["price"] >= 0)
].copy()

if df.empty:
    raise ValueError("No valid sales records remain after cleaning.")

# Calculate revenue for each record
df["revenue"] = df["quantity"] * df["price"]

# Create date-related columns
df["month"] = df["date"].dt.to_period("M")
df["day_type"] = df["date"].dt.dayofweek.map(
    lambda day: "Weekend" if day >= 5 else "Weekday"
)

print("\nData cleaning completed!")
print("Initial rows:", initial_rows)
print("Cleaned rows:", len(df))
print("Rows removed:", initial_rows - len(df))

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 6. BUSINESS QUESTION 1:
# TOP 5 PRODUCTS BY REVENUE
top_products = (
    df.groupby("product")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

print("\n" + "=" * 60)
print("1. TOP 5 PRODUCTS BY REVENUE")
print("=" * 60)
print(top_products.to_string())

# BAR CHART
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_products.values,
    y=top_products.index
)
plt.title("Top 5 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("top_products_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. BUSINESS QUESTION 2:
# REVENUE BY REGION
region_revenue = (
    df.groupby("region")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("2. REVENUE BY REGION")
print("=" * 60)
print(region_revenue.to_string())
print("Highest revenue region:", region_revenue.idxmax())

# PIE CHART
plt.figure(figsize=(8, 8))
plt.pie(
    region_revenue.values,
    labels=region_revenue.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Revenue Distribution by Region")
plt.axis("equal")
plt.tight_layout()
plt.savefig("region_revenue_pie_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 8. BUSINESS QUESTION 3:
# MONTHLY REVENUE TREND
monthly_revenue = (
    df.groupby("month")["revenue"]
    .sum()
    .sort_index()
)

print("\n" + "=" * 60)
print("3. MONTHLY REVENUE TREND")
print("=" * 60)
print(monthly_revenue.to_string())

# LINE CHART
plt.figure(figsize=(10, 6))
plt.plot(
    monthly_revenue.index.astype(str),
    monthly_revenue.values,
    marker="o",
    linewidth=2
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_revenue_line_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 9. BUSINESS QUESTION 4:
# AVERAGE ORDER VALUE
order_values = df.groupby("order_id")["revenue"].sum()
total_revenue = df["revenue"].sum()
total_orders = df["order_id"].nunique()

average_order_value = (
    total_revenue / total_orders if total_orders else 0
)

print("\n" + "=" * 60)
print("4. AVERAGE ORDER VALUE")
print("=" * 60)
print(f"Total revenue: {total_revenue:,.2f}")
print(f"Unique orders: {total_orders}")
print(f"Average order value: {average_order_value:,.2f}")

# 10. BUSINESS QUESTION 5:
# WEEKDAY VS WEEKEND SALES
day_sales = (
    df.groupby("day_type")
    .agg(
        total_revenue=("revenue", "sum"),
        total_quantity=("quantity", "sum"),
        unique_orders=("order_id", "nunique")
    )
    .reindex(["Weekday", "Weekend"], fill_value=0)
)

print("\n" + "=" * 60)
print("5. WEEKDAY VS WEEKEND SALES")
print("=" * 60)
print(day_sales.to_string())

if day_sales["total_revenue"].nunique() > 1:
    highest_sales_day = day_sales["total_revenue"].idxmax()
    print("Higher revenue day type:", highest_sales_day)
else:
    highest_sales_day = "Tie"
    print("Weekday and weekend revenue are tied.")

# OPTIONAL COMPARISON CHART
plt.figure(figsize=(8, 5))
sns.barplot(
    x=day_sales.index,
    y=day_sales["total_revenue"].values
)
plt.title("Weekday vs Weekend Revenue")
plt.xlabel("Day Type")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("weekday_weekend_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. SUMMARY OF FINDINGS
best_product = top_products.index[0]
best_region = region_revenue.index[0]
highest_month = monthly_revenue.idxmax()
lowest_month = monthly_revenue.idxmin()

findings = f"""
SALES DATA EDA - KEY FINDINGS
=============================

1. Top product by revenue: {best_product}
   Revenue: {top_products.iloc[0]:,.2f}

2. Highest revenue region: {best_region}
   Revenue: {region_revenue.iloc[0]:,.2f}

3. Highest revenue month: {highest_month}
   Revenue: {monthly_revenue.max():,.2f}

4. Lowest revenue month: {lowest_month}
   Revenue: {monthly_revenue.min():,.2f}

5. Average order value: {average_order_value:,.2f}

6. Weekday vs weekend:
   Higher revenue day type: {highest_sales_day}

7. Total revenue: {total_revenue:,.2f}
8. Total unique orders: {total_orders}
9. Valid sales records: {len(df)}

CONCLUSION
----------
This project cleans and analyzes retail sales data using
Pandas, Matplotlib, and Seaborn. The results identify top
products, regional performance, monthly revenue trends,
average order value, and weekday/weekend sales patterns.
"""

print("\n" + findings)

# 12. EXPORT RESULTS
df.to_csv("cleaned_sales_data.csv", index=False)

top_products.rename("revenue").to_csv("top_products.csv")
region_revenue.rename("revenue").to_csv("region_revenue.csv")
monthly_revenue.rename("revenue").to_csv("monthly_revenue.csv")
day_sales.to_csv("weekday_weekend_sales.csv")

with open("eda_findings.txt", "w", encoding="utf-8") as file:
    file.write(findings)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("Files created:")
print("- cleaned_sales_data.csv")
print("- top_products.csv")
print("- region_revenue.csv")
print("- monthly_revenue.csv")
print("- weekday_weekend_sales.csv")
print("- eda_findings.txt")
print("- top_products_bar_chart.png")
print("- region_revenue_pie_chart.png")
print("- monthly_revenue_line_chart.png")
print("- weekday_weekend_chart.png")