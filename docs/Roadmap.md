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
- [ ✅ ] Load & inspect: Use pandas in notebook—`df.info()`, `df.describe()`, check for nulls/duplicates.
- [ ] Clean data
- [ ] Feature engineering
- [ ] Optional: API pull—Use spotipy to fetch 50 recent tracks; merge with main dataset.
- [ ] Save processed: Export to `data/processed/train.csv` and `test.csv`. Commit data folder (or .gitignore large files, upload separately).

#### Phase 3: Exploratory Data Analysis (EDA) (1-2 Days)
- [ ] Univariate: Histograms/boxplots for features.
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


## 🧩 Phase 2: Data preprocessing

Preprocessing should be reusable by any model because You might use different splits later (e.g. 70/30, 80/20, k-fold).

In this phase, we’ll:

Acquire the dataset — load it from Kaggle or another source.

Inspect and understand the data — explore columns, check data types, and identify missing or noisy data.

Clean the dataset — handle missing values, duplicates, and outliers.

Engineer features — normalize and encode relevant columns for linear regression.


## ✅ Phase 3: EDA (Exploratory Data Analysis)
## ✅ Phase 4: Modeling
## ✅ Phase 5: Evaluation & Iteration

---

### 1️⃣ Modeling (Training Script) — src/models/train_model.py

Purpose:
This is where the machine learns.
It takes your cleaned data → splits it → fits the model → saves the trained .pkl.

You only run this when you want to (re)train your model.

**WHy:**
When you improve data, add new features, or tune hyperparameters — you’ll re-run this script to update your trained model.

---

### 2️⃣ Evaluation Script — src/models/evaluate_model.py

**Purpose:**
This checks how good your model actually is — on unseen data.

You use it after training, or later when comparing multiple models.

**Why:**
You don’t want to just train — you want to prove that your model performs well and track that performance.

---

### 3️⃣ Iteration Script — src/models/experiment_runner.py (optional)

Purpose:
Automate trying different models or parameters to find the best one.

**Why:**
Instead of manually re-running notebooks, you can loop through model types and save the one that performs best — perfect for reproducible experiments.

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
