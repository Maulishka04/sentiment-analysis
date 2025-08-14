# Sentiment Analysis of Customer Reviews with Trend Visualization

A complete end-to-end project that cleans Amazon Fine Food Reviews, trains multiple ML models, and ships an interactive **Streamlit dashboard** to analyze sentiments and visualize patterns.

> Main files in this repo: `Sentiment Analysis.ipynb` (training + preprocessing) and `app.py` (Streamlit app).  
> Large files (dataset and model artifacts) are **not included** in this repository. Follow the steps below to download or regenerate them.

---

## 🚀 Features

- **Data cleaning & labeling** for Positive/Neutral/Negative classes (from star ratings).
- **Multiple models** trained & compared: Logistic Regression, Naive Bayes, SVM, XGBoost.
- **TF-IDF vectorization** pipeline for text features.
- **Interactive dashboard** (Streamlit):
  - Model accuracy leaderboard + detailed metrics
  - Enter any review → get predicted sentiment
  - Random review tester
  - Word clouds for each sentiment
  - Sentiment distribution (pie + bar)

---

## 📂 Getting the Dataset & Model Files

Large files (`Reviews.csv`, `cleaned_reviews.csv`, `.pkl` model files, `tfidf_vectorizer.pkl`, `model_accuracies.json`) are not in this repo to keep it lightweight.

### 1) Download the dataset
You can get the **Amazon Fine Food Reviews** dataset here:  
[Kaggle Link](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

Save the original `Reviews.csv` in the same folder as the notebook.

### 2) Generate cleaned dataset & model files
Open and run **`Sentiment Analysis.ipynb`** — it will:
- Clean the dataset
- Train all models
- Save:
  - `cleaned_reviews.csv`
  - `tfidf_vectorizer.pkl`
  - `{model}.pkl` files for Logistic Regression, Naive Bayes, SVM, XGBoost
  - `model_accuracies.json`

These files must be placed in the same folder as `app.py` for the Streamlit app to work.

---

## 🗂️ Repository Structure
├── app.py

└── Sentiment Analysis.ipynb
> Large files are excluded.

---

## 📊 Dataset

- **Source**: Amazon Fine Food Reviews dataset (public Kaggle dataset).
- **Size**: 500,000+ reviews (subset used here).
- **Important columns:**
  - `Score` → converted into sentiment labels:
    - Positive (★4–5)
    - Neutral (★3)
    - Negative (★1–2)
  - `Text` → cleaned and vectorized for model training.

---

## 🔬 ML Pipeline (Notebook)

1. Load raw reviews → label sentiments from ratings.  
2. Clean text (lowercase, punctuation removal, stopwords, etc.).  
3. Split into train/test sets.  
4. Vectorize text using **TF-IDF**.  
5. Train models: **Logistic Regression**, **Naive Bayes**, **SVM**, **XGBoost**.  
6. Evaluate using Accuracy, Precision, Recall, and F1-score.  
7. Save artifacts for the Streamlit app.

---

## 🖥️ App (Streamlit) – What it shows

- **Model Accuracies**: leaderboard + classification reports
- **Analyze Review**: choose model, type custom text → prediction
- **Word Clouds**: Positive / Neutral / Negative
- **Sentiment Distribution**: pie + bar with counts

---

## ⚙️ Setup

### 1) Create and activate virtual environment

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib wordcloud matplotlib
```

### 3) Place artifacts at project root

After generating from the notebook, ensure these files are in the same folder as `app.py`:
```pgsql
cleaned_reviews.csv
tfidf_vectorizer.pkl
logistic_model.pkl
naive_bayes_model.pkl
svm_model.pkl
xgboost_model.pkl
model_accuracies.json
```

### 4) Run the app
```bash
streamlit run app.py
```

---

## 🧰 Requirements
```nginx
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
wordcloud
matplotlib
```

---

## 🙌 Acknowledgements
Amazon Fine Food Reviews dataset (Kaggle)
Scikit-learn, Streamlit, XGBoost, Matplotlib, WordCloud
