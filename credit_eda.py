"""Exploratory data analysis of loan applications merged with previous-application history."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

os.makedirs("results", exist_ok=True)


def impute(frame):
    """Median for skewed numeric columns, mean otherwise, mode for categoricals."""
    for col in frame.select_dtypes(include=np.number).columns:
        if frame[col].isnull().any():
            fill = frame[col].median() if frame[col].skew() > 1 else frame[col].mean()
            frame[col] = frame[col].fillna(fill)
    for col in frame.select_dtypes(exclude=np.number).columns:
        if frame[col].isnull().any():
            frame[col] = frame[col].fillna(frame[col].mode()[0])
    return frame


def save(name):
    plt.tight_layout()
    plt.savefig(f"results/{name}.png", dpi=150)
    plt.close()


# 1. Load data
app = pd.read_csv("data/application_data.csv")
prev = pd.read_csv("data/previous_application.csv")
print("Application shape:", app.shape)
print("Previous application shape:", prev.shape)

# 2. Handle missing values
print("Missing values before (application):", int(app.isnull().sum().sum()))
app = impute(app)
prev = impute(prev)

# 3. Aggregate previous applications to one row per customer
prev_agg = prev.groupby("SK_ID_CURR").agg(
    PREV_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    PREV_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
    PREV_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
).reset_index()

# 4. Merge
df = app.merge(prev_agg, on="SK_ID_CURR", how="left")
print("Merged shape:", df.shape)
no_history = df["PREV_CREDIT_MEAN"].isnull().mean() * 100
print(f"Customers with no previous application: {no_history:.1f}%")

# 5. Target distribution
sns.countplot(x="TARGET", data=df)
plt.title("Loan Default Distribution")
save("target_distribution")

# 6. Univariate: income distribution
plt.figure()
sns.histplot(df["AMT_INCOME_TOTAL"], bins=50, log_scale=True)
plt.title("Income Distribution (log scale)")
save("income_distribution")

# 7. Bivariate: income and previous credit vs default
plt.figure()
sns.boxplot(x="TARGET", y="AMT_INCOME_TOTAL", data=df)
plt.yscale("log")  # heavy right-skew: a few extreme incomes squash a linear axis
plt.title("Income vs Default (log scale)")
save("income_vs_default")

plt.figure()
sns.boxplot(x="TARGET", y="PREV_CREDIT_MEAN", data=df)
plt.yscale("log")
plt.title("Previous Credit vs Default (log scale)")
save("prev_credit_vs_default")

# 8. Multivariate: correlation heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
save("correlation_heatmap")

top = numeric_df.corr()["TARGET"].drop("TARGET").abs().sort_values(ascending=False).head(5)
print("\nTop 5 features by |correlation| with TARGET:\n", top)

# 9. Scale selected features
df["INCOME_SCALED"] = StandardScaler().fit_transform(df[["AMT_INCOME_TOTAL"]])
df["PREV_CREDIT_SCALED"] = StandardScaler().fit_transform(df[["PREV_CREDIT_MEAN"]])

# 10. Summary
print(f"\nDefault rate: {df['TARGET'].mean() * 100:.2f}%")
