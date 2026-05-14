# Polymarket mispricing

## Question
Are Polymarket prediction markets well-calibrated, and can systematic mispricings be identified and corrected with a model?

## Data
Pulled 5,077 resolved markets from the Polymarket Gamma API (snapshot taken before April 2026). Filtered to volume ≥ $100 and prices strictly between 0 and 1, leaving 2,355 markets for analysis. Markets priced at exactly $0 or $1 are locked-in by arbitrage and don't express real uncertainty, so they're dropped, and they'd break log loss anyway.

## Methodology
Calibration analysis run per category to find out where the miscalibration actually lives. For the mispricing model, trained two: logistic regression as a linear baseline and random forest as the main model (RF picks up the category × price interactions that a linear model can't). Used a stratified 80/20 split on `category × outcome` to keep distribution balance between train and test. Originally tried a time-based split but ran into extreme category drift: Polymarket's market mix shifted a lot over time, so categories like Geopolitics ended up in the test set with 0 markets in train.

## Key findings
1. Polymarket is well-calibrated below p=0.6 but systematically overconfident above that, mostly driven by sports and crypto markets.
2. Random forest achieved a Brier score of 0.126 on a stratified holdout of 471 markets vs the market's 0.164 (23% improvement).
3. The improvement was concentrated in sports (~16% improvement) and crypto (~42% improvement). Categories that were already calibrated were untouched.

## Reproducibility
Notebook execution order: `data_collection.ipynb` → `calibration_analysis.ipynb` → `mispricing_model.ipynb`. The `resolved_markets.csv` file is committed, so the analysis notebooks run directly without re-pulling from the API.

Dependencies: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`. Install with:

`pip install pandas numpy scikit-learn statsmodels matplotlib`

Note: re-running `data_collection.ipynb` pulls live Polymarket data and will produce different results. The committed CSV is a snapshot from April 2026.