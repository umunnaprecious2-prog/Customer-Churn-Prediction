# Customer Churn Prediction — Execution Plan (per instruct.md)

## Context

[instruct.md](../../../Documents/AI%20PROJECTS/Customer%20Churn%20Prediction/instruct.md) lays out a full data-science + business-analysis assignment: explore the data, prepare it, build a baseline and advanced models, evaluate them properly, translate results into a financial risk score, reason about prediction-error costs, and produce a set of named deliverables (model, reports, ranked list, dashboard).

We've already done the exploration step conversationally (not yet saved to a file) and found several data issues that directly affect how faithfully we can execute the later steps:

- 300 exact duplicate rows (~9.5%)
- `Age` is redundant — fully determined by `Age Group`
- `Status` (1=active/2=non-active) correlates so strongly with `Churn` (5% vs 47%) that it's a likely leakage feature — it isn't in instruct.md's list of variables to analyze
- `Charge Amount` is a 0–10 ordinal tier, not a currency figure
- 132 customers (4.2%) are fully dormant (zero usage, `Customer Value`=0) — this will silently zero out their Risk Score regardless of churn probability
- `Frequency of use` / `Seconds of Use` are highly collinear (r=0.95)
- No missing values; no impossible values otherwise

The goal of this plan is to turn instruct.md's checklist into an ordered, concrete set of scripts/artifacts, with the above findings baked in as explicit, documented decisions (not silently applied) — since the last planning attempt, we deferred committing on `Status` and on how far to take the app. This plan stays scoped to **exactly what instruct.md asks for**: a model + business analysis + dashboard, not the multi-service platform discussed separately (that remains a deferred/future idea, not part of this plan).

`xgboost`, `lightgbm`, `shap`, and `imblearn` are **not** currently installed (`sklearn`, `joblib`, `matplotlib` are). instruct.md requires XGBoost and LightGBM, so installing them is step 1.

## Decisions this plan bakes in (flag if you disagree before I start building)

- **Duplicates**: drop the 300 exact-duplicate rows before any split/training.
- **`Age`**: drop it, keep `Age Group` (same information, `Age` is not real age).
- **`Status`**: excluded from model features by default (leakage risk, not in instruct.md's variable list) but kept in the cleaned dataset for reference/EDA commentary. Reported clearly in the EDA writeup so it's an informed, reversible choice, not a silent one.
- **Dormant customers (`Customer Value`=0)**: kept, flagged with an `is_dormant` column, not dropped. Explicitly called out in the risk-scoring deliverable since Risk Score collapses to 0 for them regardless of churn probability — we'll present them as a separate "high churn-probability, currently low value" watchlist rather than let them silently disappear from the ranked list.
- **Collinearity** (`Frequency of use` vs `Seconds of Use`): keep both (instruct.md explicitly asks to analyze "frequency of service usage"), rely on L2-regularized Logistic Regression to handle it, and note the correlation as a caveat when discussing LR coefficients.
- **Class imbalance** (15.7% churn): handle via `class_weight='balanced'` (sklearn/XGBoost `scale_pos_weight`/LightGBM `is_unbalance`) rather than SMOTE, to avoid synthesizing rows on top of an already small (~2,850-row) dataset. No new dependency needed.
- **FP/FN cost assumptions**: instruct.md asks to "recommend an appropriate classification threshold considering business objectives" but the dataset has no currency-denominated campaign cost. We'll use `Customer Value` as the FN cost proxy (money lost if a real churner is missed) and a small, clearly-labeled assumed flat cost per contacted customer for FP (e.g. a retention-offer cost), presented as an adjustable parameter — not a hard-coded fact — so the business can plug in their real numbers later.

## Proposed structure

Per explicit user request: the plan itself gets saved as a project file, and the whole process is carried out **step by step inside a single Jupyter notebook** (code + markdown narration together, so the reasoning and the code stay attached to each other) rather than as separate `.py` scripts.

```
Customer Churn Prediction/
├── Customer Churn.csv                    (raw, untouched)
├── instruct.md
├── PLAN.md                                (this plan, saved to the project — step 0)
├── Customer_Churn_Analysis.ipynb          (the whole process, step by step, code + markdown)
├── data/
│   └── customer_churn_cleaned.csv         (written by the notebook's cleaning section)
├── models/
│   └── best_model_pipeline.joblib         (fitted preprocessing + best model, saved from the notebook)
└── outputs/
    ├── model_comparison.csv
    ├── high_risk_customers.csv
    ├── dormant_watchlist.csv
    └── feature_importance.csv
```

A dashboard (final instruct.md deliverable) will be published as a Claude Artifact — an interactive HTML page pulling from `outputs/` (KPIs, churn-rate breakdowns by the variables instruct.md names, ranked risk table, threshold trade-off chart, feature importance chart) — built with the `dataviz` skill once the notebook has produced the underlying CSVs/metrics.

`clean_data.py`, drafted earlier as a standalone script, gets folded into the notebook's first section instead of staying separate, so the whole pipeline lives in one place the user can read top to bottom.

## Step-by-step execution order

0. **Save this plan** to `PLAN.md` in the project root.
1. **Install missing packages**: `xgboost`, `lightgbm` (required by instruct.md). Hold off on `shap`/`imblearn` unless we decide we want them. Also register a Jupyter kernel for this environment if one isn't already available.
2. **Create `Customer_Churn_Analysis.ipynb`** and build it out section by section, each section = a markdown cell explaining *what* and *why*, followed by the code cell(s) that do it, followed by a look at the actual output (table/plot) before moving on — not one big code dump. Sections, in order:
   1. **Setup & raw load** — imports, load `Customer Churn.csv`, first look.
   2. **Data cleaning** — normalize column names, drop the 300 duplicates, drop redundant `Age` (keep `Age Group`), add `is_dormant` flag; verify row counts/churn rate before/after; save `data/customer_churn_cleaned.csv`.
   3. **EDA** — distributions and churn-rate breakdowns for the instruct.md-named variables (call failures, subscription length, customer value, charge amount, complaints, age group, frequency of use/SMS), correlation table, and an explicit written note on the `Status` leakage decision.
   4. **Preprocessing** — `ColumnTransformer` (scaling for LR, pass-through for already-encoded ordinals), stratified train/test split; `Status` excluded from the feature set here.
   5. **Modeling** — Logistic Regression baseline (`class_weight='balanced'`), then Random Forest, XGBoost, LightGBM, each with class-imbalance handling; 5-fold stratified CV comparison plus final held-out test fit.
   6. **Evaluation** — Precision/Recall/F1/ROC-AUC + confusion matrix per model, comparison table saved to `outputs/model_comparison.csv`, written model selection rationale.
   7. **Threshold analysis** — cost sweep using FN=`Customer Value` / FP=assumed flat cost (clearly labeled as an adjustable assumption), recommended threshold with reasoning (instruct.md's "Important Consideration").
   8. **Risk scoring** — `Risk Score = P(churn) × Customer Value` on the full dataset, ranked `outputs/high_risk_customers.csv`, separate `outputs/dormant_watchlist.csv` for zero-value-but-at-risk customers.
   9. **Feature importance** — LR coefficients (standardized) + tree importances → `outputs/feature_importance.csv`, written interpretation.
   10. **Business analysis** — estimated revenue at risk, threshold recommendation, FP/FN cost discussion, narrative retention recommendations tied to the top churn drivers.
   11. **Save the final pipeline** — `models/best_model_pipeline.joblib`.
3. **Dashboard artifact**: interactive HTML built from the notebook's `outputs/` CSVs and business-analysis findings, published via the Artifact tool (dataviz skill) — the final instruct.md deliverable.

## Verification

- After step 2: confirm `data/customer_churn_cleaned.csv` has 2,850 rows, no duplicates, churn rate ≈15.7%, and `age` column absent.
- After step 5–6: confirm all 4 models produce non-degenerate confusion matrices (i.e., no model collapses to predicting a single class) and that CV scores are reasonably stable across folds (small dataset, so variance across folds is itself worth reporting).
- After step 8: confirm the ranked `high_risk_customers.csv` sums of `Customer Value` are internally consistent with the `business_analysis.md` revenue-at-risk figure.
- Final: open the published dashboard artifact and visually confirm KPIs match the underlying CSVs.

## Open item

The `Status` exclusion and the FP/FN cost assumptions above are defaults chosen to keep this plan moving — flag now if you want either handled differently before I start building.
