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
- [ ] Download dataset: Kaggle "Spotify Songs Dataset" (114K rows CSV). Save to `data/raw/`.
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

