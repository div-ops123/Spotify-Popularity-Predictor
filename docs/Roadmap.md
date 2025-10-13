### Project Build Checklist: Spotify Song Popularity Predictor


#### Phase 1: Setup & Project Structure (1 Day)
- [ ✅ ] Create GitHub repo: Name it "spotify-popularity-predictor" (public). Add .gitignore (for __pycache__, .env, datasets).
- [ ✅ ] Initialize Git: `git init`, add README.md with project overview (problem, tech stack: scikit-learn, pandas, Streamlit), and license (MIT).
- [ ✅ ] Set up environment: Create `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, spotipy). Use venv/Poetry.
- [ ✅ ] Define structure:
  ```
  spotify-popularity-predictor/
  ├── data/          # Raw/processed CSVs
  ├── notebooks/     # EDA/modeling Jupyter
  ├── src/           # Scripts (data_prep.py, model.py, app.py)
  ├── tests/         # Pytest files
  ├── docs/          # README, diagrams
  ├── Dockerfile     # For deployment
  └── requirements.txt
  ```
- [ ✅ ] Commit initial structure: Branch "main", push.

#### Phase 2: Data Acquisition & Preparation (1-2 Days)
- [ ✅ ] Download dataset: Kaggle "Spotify Music Dataset". Save to `data/raw/`.
- [ ] Load & inspect: Use pandas in notebook—`df.info()`, `df.describe()`, check for nulls/duplicates.
- [ ] Clean data: Handle missing (drop/fill), outliers (e.g., clip duration >10min), encode categoricals (pd.get_dummies for key/mode).
- [ ] Feature engineering: Scale numerics (StandardScaler), create interactions (e.g., energy * valence), split train/test (80/20, stratify on popularity bins).
- [ ] Optional: API pull—Use spotipy to fetch 50 recent tracks; merge with main dataset.
- [ ] Save processed: Export to `data/processed/train.csv` and `test.csv`. Commit data folder (or .gitignore large files, upload separately).

#### Phase 3: Exploratory Data Analysis (EDA) (1-2 Days)
- [ ] Univariate: Histograms/boxplots for features (e.g., seaborn.distplot for danceability); target dist (popularity skewed? Log-transform?).
- [ ] Bivariate: Correlation heatmap (sns.heatmap), scatterplots (e.g., tempo vs. popularity colored by genre).
- [ ] Multivariate: Pairplot for top 5 features; groupby genre for insights (e.g., avg popularity by explicit).
- [ ] Insights doc: Notebook section with 3-5 key findings (e.g., "Valence >0.7 correlates with 20% higher pop"). Viz saved as PNGs.
- [ ] Tie to DS&A: Sort features by correlation; use sets for unique values.
- [ ] Commit notebook: Push to `notebooks/eda.ipynb`.

#### Phase 4: Modeling (2 Days)
- [ ] Baseline: Simple LinearRegression from sklearn; fit on train, predict test.
- [ ] Advanced: Multiple linear (all features); PolynomialFeatures(degree=2) for non-linearity.
- [ ] Feature selection: Use SelectKBest or recursive elimination for top 5-7 (e.g., drop low-variance like mode).
- [ ] Train: `model.fit(X_train, y_train)`; handle assumptions (check multicollinearity with VIF).
- [ ] Save model: Joblib.dump to `models/popularity_model.pkl`.
- [ ] Commit: `src/model.py` with train function.

#### Phase 5: Evaluation & Iteration (1 Day)
- [ ] Metrics: R², MAE, RMSE on test set; compare models in table (e.g., baseline vs. poly).
- [ ] Viz: Actual vs. predicted scatter, residual plot (for linearity/homoscedasticity).
- [ ] Cross-val: 5-fold CV score; discuss overfitting (e.g., if train R² >> test).
- [ ] Iterate: Tune (e.g., Ridge for regularization); re-run if R² <0.5.
- [ ] Tests: Pytest in `tests/test_model.py` (e.g., assert R² >0.5).
- [ ] Commit: Update notebook with results; add metrics to README.

#### Phase 6: Frontend & Integration (1-2 Days)
- [ ] Build UI: Streamlit app in `src/app.py`—sliders for features, predict button, output score + bar chart of contributions.
- [ ] Integrate: Load model/data; add API pull option for real-time features.
- [ ] Enhancements: Suggestion logic (e.g., "Increase danceability by 0.1 for +5 points").
- [ ] Local test: `streamlit run app.py`; debug edge cases (e.g., invalid inputs).
- [ ] Commit: Include app script.

#### Phase 7: Deployment & Polish (1 Day)
- [ ] Dockerize: Basic Dockerfile (FROM python:3.9, COPY ., pip install -r requirements).
- [ ] Deploy: Push to Heroku/Streamlit Cloud; set env vars if needed (e.g., Spotify API key).
- [ ] Monitor: Add basic logging; test live predictions.
- [ ] Polish: Update README (setup instructions, demo GIF, architecture diagram via Draw.io); add badges (e.g., coverage).
- [ ] Final commit: Merge branches, tag v1.0, push.

#### Phase 8: Documentation & Share (Ongoing)
- [ ] Portfolio prep: Blog post or LinkedIn draft with key visuals/metrics.
- [ ] Self-review: Run linters (black), full tests; simulate recruiter questions (e.g., "Why linear over random forest?").
- [ ] Backup: Archive dataset link in README.


🧩 Phase 2 Overview: Data Acquisition & Preparation

In this phase, we’ll:

Acquire the dataset — load it from Kaggle or another source.

Inspect and understand the data — explore columns, check data types, and identify missing or noisy data.

Clean the dataset — handle missing values, duplicates, and outliers.

Engineer features — normalize and encode relevant columns for linear regression.


---

# Phase 2 — Data Acquisition & Preparation (Production-ready checklist & steps)


## 4) Quick validation steps (very important)

Run these checks in your ingestion pipeline and fail early:

* Row count sanity (nonzero, not huge unexpected jump).
* Uniqueness of `track_id`.
* Required columns exist and types match schema.
* No corrupt rows (e.g., strings in numeric columns).
* Basic ranges: `0 <= popularity <= 100`, `0 <= danceability <= 1`, `tempo > 0`.

---

## 5) Cleaning plan (step-by-step)

1. **Deduplicate** on `track_id`. Keep latest `release_date` if duplicates differ.
2. **Handle missing values**:

   * If a numeric audio feature missing for <1% rows → impute with median.
   * If a column missing >30% → drop or decide whether to collect via API.
   * Document every imputation in `data_prep_log.md`.
3. **Parse date fields**: `release_date` → datetime; derive `release_year`, `age_days`.
4. **Feature consistency**: convert booleans, cast dtypes (float32 for memory).
5. **Outlier handling**:

   * For `duration_ms`, `tempo`, or `loudness` check extreme tails. Do NOT blindly clip; prefer log transforms where appropriate (e.g., duration_ms).
6. **Target engineering**:

   * Decide: predict raw `popularity` (0–100, regression) or a transformed target? Usually keep raw, but consider modeling logit or scaled variant if distribution is skewed.
7. **Save cleaned dataset**: `data/clean/spotify_clean_v1.parquet` and write `data/clean/README.md` with counts & transformations.

---

## 6) Feature engineering ideas (high-value, production-minded)

* **Temporal features**: `release_year`, `days_since_release`, `is_new_release` (release within 90 days).
* **Audio ratios**: `vocals_ratio = 1 - instrumentalness` (if meaningful).
* **Normalized loudness**: `loudness_db` is negative dBFS; consider rescaling.
* **Interaction features**: `danceability * energy` (maybe predictive).
* **Binned tempo**: tempo buckets (slow, mid, fast) using domain knowledge.
* **Categorical encodings**: plan for `one-hot` or target encoding for `genre` or `artist_popularity_bucket` but beware leakage (use train-only stats).
* **Artist-level aggregated features**: avg artist popularity, #tracks, average energy — compute on train only to avoid leakage.
* **Text features (optional)**: Track name embeddings, lyrics (if available), but keep initial model simple.

Document feature transformation code in notebook + `features/feature_defs.yaml`.

---

## 7) Avoid target leakage (critical)

* Do not use future information w.r.t. prediction moment: e.g., if popularity was measured over a window after release, ensure features are available at prediction time.
* When computing artist historical averages, compute from data that would be available prior to the track's release (time-split!).

---

## 8) EDA & diagnostics (what to run now)

* **Distribution**: histogram of `popularity`. Is it skewed or multi-modal?
* **Pairwise correlations**: `df.corr()` heatmap for float features.
* **Feature vs target plots**: scatter `popularity` vs `danceability`, `energy`, `tempo`.
* **Residual-analogue**: not yet, but you can fit a quick baseline and inspect residuals.
* **Missingness matrix**: `missingno` library.
* **Multicollinearity**: compute VIF after preparing numeric features (do this now).

  * Use `statsmodels` VIF implementation.
* **Document everything** in `notebooks/02_EDA.ipynb` as a clean narrative.

---

## 9) Multicollinearity checks (you asked earlier)

* Compute VIF for numeric features.

* If VIF > 5 (or 10) for a feature → consider:

  * dropping one of correlated features,
  * combining them,
  * or plan to use Ridge/ElasticNet in modeling.

---

## 10) Train/validation/test split (data practice)

* Use **time-aware splitting** if you plan to predict future popularity (train on earlier releases, test on later releases). If not, standard stratified random split by popularity buckets.
* Example (time split):

  * Train: releases <= 2022-06-30
  * Val: 2022-07-01 to 2022-09-30
  * Test: after 2022-10-01
* Save splits manifest `splits/split_v1.json` with indices and counts.

---

## 11) Data pipelines & reproducibility

* Build simple pipeline with `make` or Prefect/Airflow:

  * `ingest -> validate -> clean -> featurize -> save`
* Keep transformations as code (not manual notebook steps) in `src/data/`.
* Write unit tests for transformations (e.g., `tests/test_ingest.py`) ensuring schema consistency.
* CI: run data validation and the pipeline on PRs.

---

## 14) Quick evaluation plan for later (so you prepare data accordingly)

* Metrics: **RMSE**, **MAE**, **R²** (main), and consider **Spearman rank** if you care about relative ranking of songs.
* Prepare to evaluate both overall and per-segment (new releases vs catalog, genre slices).
* Keep a holdout test set you touch only once.

---

## 15) Documentation checklist to finish Phase 2

* `DATA_SCHEMA.md` — ✅
* `ingest_manifest.json` — ✅
* `data_prep_log.md` describing all transforms — ✅
* `notebooks/02_EDA.ipynb` with visuals and VIF results — ✅
* `data/clean/spotify_clean_v1.parquet` stored & versioned — ✅

---

## TL;DR — Minimal tasks to complete Phase 2 (do these now)

1. Download raw Kaggle dataset and save as immutable raw parquet.
2. Create `DATA_SCHEMA.md` and ingest manifest.
3. Run cleaning pipeline (dedupe, parse dates, median impute) -> save cleaned parquet.
4. Run EDA notebook: hist(popularity), corr heatmap, missingness matrix.
5. Compute VIF & document top problematic features.
6. Create time-aware train/val/test splits and save split manifest.
7. Add data tests and version data with DVC or S3 + git metadata.

---
