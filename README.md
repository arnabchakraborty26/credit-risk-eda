# Credit Risk Analysis (Exploratory Data Analysis)

Exploratory data analysis of loan applications merged with each customer's previous-application history, to understand what distinguishes customers who default.

## Data
Home Credit-style loan data, available on Kaggle (not included here because the files are 166 MB and 405 MB):
- `application_data.csv`: 307,511 applications, 122 columns, with the `TARGET` default flag.
- `previous_application.csv`: 1,670,214 earlier applications, 37 columns.

Download both files and place them in `data/`.

## Approach
1. **Missing values:** median for skewed numeric columns (skew > 1), mean for the rest, mode for categoricals.
2. **Feature engineering:** aggregate previous applications per customer (mean credit, annuity and payment count) and left-join onto the applications.
3. **Analysis:** target distribution, income distribution, income and previous credit vs default (log-scaled boxplots, since both are heavily right-skewed), and a correlation heatmap.
4. **Transformation:** standardize income and previous credit.

## Key findings
- **Class imbalance:** only 8.07% of applicants defaulted, so plain accuracy would be a misleading metric for any model built on this data.
- **Missing history:** 5.4% of customers have no previous application, so their aggregated features stay empty after the merge.
- **Strongest correlates of default** are `EXT_SOURCE_2`, `EXT_SOURCE_3` and `EXT_SOURCE_1` (|r| around 0.10 to 0.16), followed by `DAYS_BIRTH`. All correlations with `TARGET` are weak, so no single feature separates defaulters well.
- Median income is slightly lower among defaulters, but the distributions overlap heavily.

Plots are in `results/`, console output in `results/eda_output.txt`.

## Run
```bash
pip install -r requirements.txt
python credit_eda.py
```
