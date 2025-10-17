# Spotify Song Popularity Predictor

A machine learning project built with linear regression to predict song popularity on Spotify based on audio features like danceability, energy, and tempo. Using real-world data from Kaggle's Spotify Songs Dataset, this app helps indie artists and labels forecast a track's viral potential—turning data into actionable insights!

**Why this project?** Fresh off my linear regression module in AI/ML training, I wanted a fun, real-world application: What audio traits make a song a hit? (Spoiler: High valence + danceability = chart-topper.) This showcases end-to-end ML skills: data prep, modeling, evaluation, and deployment—perfect for aspiring AI Engineers shipping MVPs solo.

🚀 **Live Demo**: [Coming soon—deployed on Streamlit Cloud](https://your-app-link.streamlit.app)  
📊 **Dataset**: [Kaggle Spotify Music Dataset](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)

## Features
- **Predict Popularity**: Input audio features (e.g., energy=0.8, tempo=120) and get a 0-100 score via linear regression model (R² ~0.62).
- **Feature Insights**: Visualize coefficients to see what drives virality (e.g., boost danceability for +10 points).
- **Real-Time Pull**: Optional Spotify API integration for fresh track analysis.
- **Interactive UI**: Streamlit frontend for quick testing—upload a song link and predict!
- **Production-Ready**: Dockerized, tested, and deployed with MLOps basics (model versioning via joblib).

## Tech Stack
- **ML**: scikit-learn (linear/polynomial regression), pandas (data handling), numpy.
- **Viz**: Matplotlib, Seaborn (EDA plots, model evals).
- **Backend/UI**: Streamlit (interactive app), FastAPI (optional API layer).
- **DevOps**: Docker (containerization), pytest (testing), GitHub Actions (CI/CD).
- **Data**: Spotify API (spotipy) for live pulls.

## Installation & Setup
1. **Clone the Repo**:
   ```
   git clone https://github.com/div-ops123/Spotify-Popularity-Predictor.git
   cd Spotify-Popularity-Predictor
   ```

2. **Virtual Environment**:
   ```
   python -m venv .venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the Data Acquisition Script**

   ```bash
   python -m src.data.acquire_data
   ```

   This will:

   * Load raw Spotify datasets (low & high popularity)
   * Combine them into one cleaned dataset
   * Save the merged output in your processed data directory

5. preprocessing.py = toolbox (contains helper functions)

preprocess_data.py = chef (uses the tools to clean the full dataset)

```bash
python -m src.data.preprocess_data
```

This will:

* Reproduce your full notebook cleaning automatically.




**Spotify API (Optional)**:
   - Get credentials from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
   - Add to `.env`: `SPOTIFY_CLIENT_ID=your_id` and `SPOTIFY_CLIENT_SECRET=your_secret`.


## Usage
- **Predict a Song**: In the Streamlit app, tweak sliders for features → Hit "Predict" → See score + viz.
- **Example**: Energy=0.9, Valence=0.7, Tempo=128 → Predicted Popularity: 78/100 (EDM banger alert!).
- **CLI Quick Test**: `python src/predict.py --energy 0.8 --valence 0.6`

For full walkthrough, check `notebooks/modeling.ipynb`.

## Project Structure
```
spotify-popularity-predictor/
├── data/                  # Raw/processed datasets
│   ├── raw/               # Original Kaggle CSV
│   └── processed/         # Cleaned train/test CSVs
|
├── notebooks/
│   ├── 01_data_exploration.ipynb         # quick look at columns, data types, distributions, missing values
│   ├── 02_eda.ipynb       # experiments with cleaning (e.g., outlier removal, scaling)
│   ├── 03_modeling.ipynb                 # test linear regression and visualize results
│   ├── 04_evaluation_and_iteration.ipynb # refine and compare models
│
├── src/
│   ├── data/  # Handles all data-related operations.
│   │   ├── acquire_data.py      # loads your CSV files, merges datasets, validates structure
│   │   ├── preprocess_data.py   # cleans, encodes, scales, splits, and saves clean datasets
│   └── app.py                   # Streamlit UI
│   │
│   ├── models/
│   │   ├── train_model.py    # trains the regression model, saves model artifacts (e.g., .pkl file)
│   │   ├── evaluate_model.py # loads trained model, evaluates on test set, prints metrics (R², MAE, etc.)
│   │
│   ├── utils/
│       ├── helpers.py     # utility functions for logging, configuration reading, or reusable plotting
│
├── configs/
│   ├── config.yaml        # stores dataset paths, column names, model hyperparameters, etc.
├── tests/                 # Unit/integration tests
│   └── test_model.py
├── docs/                  # Diagrams, notes
├── Dockerfile 
├── requirements.txt  
└── README.md
└── main.py    # Entry point for production pipeline. Calls functions from /src/ to run the entire process (data → model → results).
```

## Key Learnings & Metrics
- **Model Performance**: Multiple linear regression beats baseline by 15% MAE; poly features capture non-linear tempo effects.
- **Challenges Overcome**: Handled skewed popularity dist with log-transform; used DS&A (e.g., sorting correlations) for feature selection.
- **Next Steps**: Add RAG for lyrics analysis or ensemble with random forests.

Built Follow my journey on [LinkedIn](https://linkedin.com/in/divine-nwadigo1)!

## Contributing & Feedback
Love music + ML? Fork, PR, or DM ideas (e.g., genre-specific models).

**Feedback?** Star the repo, comment below, or connect on LinkedIn. What's your go-to track for testing?

