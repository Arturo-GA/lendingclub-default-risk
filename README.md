# LendingClub default risk — credit-admission PD model

[![tests](https://github.com/Arturo-GA/lendingclub-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/Arturo-GA/lendingclub-default-risk/actions/workflows/ci.yml)

Probability-of-default model for P2P loans (LendingClub, 2010–2018) built the way a bank validates a credit-risk model before production: **origination-only features with an explicit leakage guard, out-of-time validation, banking metrics (AUC / Gini / KS / PSI), an auditable WoE scorecard, a cost-based cut-off and SHAP explainability.**

Test AUC **0.715** (LightGBM) on loans originated after the training window, in line with the 0.70–0.73 the literature reports for origination-only features on this dataset. Anything far above that almost always means post-origination information leaked in, and `scripts/leakage_ablation.py` shows it on the same data (with `last_pymnt_amnt` the AUC approaches 1).

## Results

<!-- RESULTS:START — this section is rewritten by scripts/train.py from reports/metrics/summary.md; do not edit by hand -->
LendingClub pricing variables (int_rate, grade): **included**.  
Default rate in test: **14.91%**. Threshold chosen on validation with cost FN:FP = 5:1.

| Model         |    AUC |   Gini |     KS |   PR-AUC |   Brier |   AUC train | Threshold | Approval     | Default (approved)  |
|:--------------|-------:|-------:|-------:|---------:|--------:|------------:|---------:|:-------------|:--------------------|
| lightgbm      | 0.7151 | 0.4302 | 0.3137 |   0.2986 |  0.1991 |      0.7334 |     0.51 | 65.8%        | 9.07%               |
| xgboost       | 0.7139 | 0.4278 | 0.3108 |   0.2966 |  0.1986 |      0.7373 |     0.53 | 69.1%        | 9.54%               |
| random_forest | 0.7043 | 0.4086 | 0.2936 |   0.2861 |  0.1924 |      0.7271 |     0.49 | 66.9%        | 9.55%               |
| logreg        | 0.7041 | 0.4081 | 0.298  |   0.2832 |  0.2004 |      0.6951 |     0.49 | 65.9%        | 9.44%               |
| scorecard     | 0.704  | 0.4079 | 0.2959 |   0.2835 |  0.2012 |      0.6938 |     0.51 | 66.1%        | 9.47%               |

AUC train vs test shows the degree of overfitting. `Approval` and `Default (approved)` describe the portfolio that results from rejecting everything above the threshold.

## Feature stability (PSI train → test)

33 of 64 features above 0.1. The 10 with the largest drift:

| feature                    |   psi | status    |
|:---------------------------|------:|:----------|
| num_bc_tl                  | 0.988 | unstable |
| num_tl_op_past_12m         | 0.974 | unstable |
| pct_tl_nvr_dlq             | 0.97  | unstable |
| total_il_high_credit_limit | 0.968 | unstable |
| mo_sin_old_rev_tl_op       | 0.968 | unstable |
| num_actv_rev_tl            | 0.968 | unstable |
| total_rev_hi_lim           | 0.967 | unstable |
| num_rev_tl_bal_gt_0        | 0.966 | unstable |
| num_op_rev_tl              | 0.966 | unstable |
| num_actv_bc_tl             | 0.964 | unstable |
<!-- RESULTS:END -->

Figures and tables in `reports/`: ROC/PR curves, calibration, KS, default rate by decile, PSI stability and SHAP (plot labels are in Spanish).

## Data

Public LendingClub extract 2007–2018Q4 (Kaggle, `wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz`, ~1.6 GB, 2.26 M rows × 151 columns). Not versioned; see [`data/README.md`](data/README.md).

**Target:** `default = 1` if `loan_status == "Charged Off"`, 0 if `Fully Paid`. Current loans are excluded. In credit risk the positive class is the event to detect.

**Filters:**
- Originated from 2010 onwards (before that LendingClub was a different company in volume and policy).
- **Maturity:** only loans whose term (36 or 60 months) had expired by the extract's close (2018-12). Without this filter, the 2016–2018 vintages contain only loans that resolved *early* (prepayments and early defaults), which biases the default rate.

That leaves about **800 k loans** with a default rate near **15–20 %**.

**Features:** ~60 numeric and 6 categorical variables known at origination, in three groups: loan terms, applicant profile and credit bureau. They are declared as a **whitelist** (`config.RAW_FEATURES`); any column not listed is excluded by default.

## Design decisions

**Leakage.** The dataset carries ~40 columns describing the *life* of the loan (`last_pymnt_amnt`, `total_rec_prncp`, `recoveries`, `last_fico_range_*`, hardship, settlement…). They are the outcome, not predictors. Each one is documented in `config.LEAKAGE_COLUMNS`, and `features.assert_no_leakage` fails the training run if any reaches the feature matrix. `scripts/leakage_ablation.py` trains the same model with and without them to quantify the effect.

**Out-of-time validation.** Split by origination date: oldest 70 % trains, next 15 % validates (early stopping, tuning, threshold), most recent 15 % is the test set and is not touched until the end. A random split would mix vintages and overstate performance.

**No resampling.** The imbalance (≈ 80/20) is handled with `class_weight` / `scale_pos_weight`. The test set keeps the real distribution, so the approval and default rates reported are the ones the business would see.

**Preprocessing inside the `Pipeline`.** Imputation, missing indicators, scaling, one-hot and WoE binning are fitted on train only. Missing values are informative (LendingClub added the bureau variables in 2012), so they are kept as indicators instead of dropping rows or columns.

**Banking metrics, not accuracy.** With 80 % good loans, approving everyone gives accuracy 0.80. We report AUC, Gini, KS, PR-AUC, Brier and calibration, plus a **decile table** ("if I reject the worst-scored 20 %, what fraction of defaults do I avoid and how many good customers do I lose?").

**Cost-based threshold.** An approved default costs capital; a rejected good customer costs margin. With a configurable ratio (5:1 by default, `config.COST_*`) the threshold is chosen on validation and the resulting portfolio is reported.

**Stability (PSI).** For every feature the PSI between train and test is computed. Features with PSI > 0.25 changed distribution between vintages and are candidates for review even if the AUC looks fine. In this dataset the strongest drift comes from the missingness pattern of the bureau variables introduced in 2012.

**Models.** Logistic regression (baseline), **scorecard** (logistic regression on WoE, the auditable standard in banking), Random Forest, XGBoost and LightGBM with early stopping. Tuning with Optuna on the temporal validation set. Explainability with SHAP.

**LendingClub pricing.** `int_rate`, `grade` and `sub_grade` are legitimate at origination but encapsulate the platform's internal model. `--exclude-pricing` reports the model without them (ablation).

## Reproduce

```bash
git clone https://github.com/Arturo-GA/lendingclub-default-risk && cd lendingclub-default-risk
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Put accepted_2007_to_2018Q4.csv.gz in data/raw/
# 2. Processed dataset (parquet, ~2 min; reading the CSV needs ~6 GB of RAM)
python scripts/make_dataset.py

# 3. Train the base models and generate the reports
python scripts/train.py

# 4. Tuning, explainability, leakage check and slide deck
python scripts/tune.py --model lightgbm --trials 40
python scripts/train.py --params reports/metrics/best_params.json
python scripts/explain.py --model lightgbm
python scripts/leakage_ablation.py
python scripts/make_deck.py

# Tests (synthetic data, no CSV needed)
pytest -q
```

Or all at once: `make all` (dataset → training → SHAP → leakage ablation → deck). `scripts/train.py` rewrites the *Results* section of this README with the real metrics; it never does so with synthetic data.

Quick run without the real dataset: `python tests/synthetic.py && python scripts/make_dataset.py --raw data/raw/synthetic.csv.gz && python scripts/train.py`. GitHub CI runs exactly that on every push.

## Structure

```
src/lcrisk/
  config.py      target, time window, feature whitelist, leakage blacklist, costs, PSI thresholds
  data.py        selective column loading, target, maturity filter, temporal split
  features.py    feature engineering + leakage guard
  selection.py   WoE / IV, Cramér's V, correlation pruning, WoEEncoder (scorecard)
  pipeline.py    ColumnTransformer + models, early stopping
  evaluate.py    AUC, Gini, KS, PR-AUC, Brier, calibration, deciles, cost threshold, PSI
  tune.py        Optuna on the temporal validation set
  explain.py     SHAP
scripts/         make_dataset · train · tune · explain · leakage_ablation · make_deck
notebooks/       01_eda · 02_resultados (interpret, do not train)
tests/           leakage, data, pipeline, WoE/PSI (with a synthetic generator)
reports/         metrics/*.json|md|csv · figures/*.png · presentacion.pptx
.github/         CI: tests + smoke run
```

## Limitations and next steps

- Models PD only; pricing or provisioning would need LGD/EAD.
- No per-vintage validation (*vintage curves*) or continuous monitoring; the train→test PSI is the first step.
- Selection bias: only loans approved by LendingClub are observed (*reject inference* pending).
- The scorecard uses quantile binning; optimal binning with a monotonicity constraint would improve its interpretability.

## Presentation

`python scripts/make_deck.py` builds `reports/presentacion.pptx` (11 slides, Spanish) from the reports: every number and figure comes from `reports/`, so the deck always matches the last run. Authors and programme are read from `config.py`.

## Authors

Rosario Sánchez Mosquipa · Ruth Velarde Retamozo · Juan Gutiérrez Aguilar · Katherine Mauri Córdova · Densel Castillón Medina

Capstone project of the Data Analytics Diploma, Pontificia Universidad Católica del Perú (2025). Data: LendingClub 2007–2018Q4 via Kaggle (`wordsforthewise/lending-club`).
