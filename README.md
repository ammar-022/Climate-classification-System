# AI-Based Climate Change Article Classification System

An NLP and Machine Learning system that classifies English climate-related articles into six categories and shows a probability for every category in a Streamlit dashboard.

## 1. Project introduction
The user pastes an English article or paragraph. A saved scikit-learn pipeline (TF-IDF + classifier) predicts the probability of each of six categories and the highest-probability category is shown as the final prediction.

## 2. Project objectives
- Build a supervised text-classification pipeline on the `SciDCC.csv` dataset.
- Compare three classical models fairly (cross-validation on the training set only).
- Evaluate the chosen model once on an untouched test set.
- Provide a Streamlit app that loads the saved model (no retraining) and displays genuine `predict_proba()` percentages.

## 3. Dataset description
`SciDCC.csv` contains 11,539 ScienceDaily news articles in 20 categories, with columns `Date`, `Link`, `Title`, `Summary`, `Body`, `Category`, `Year`. Missing values: only `Body` (5 records). No fully duplicated rows.

## 4. Six selected categories
Only these categories are used; all others are removed. Actual counts equal the expected counts.

| Category | Records |
|---|---:|
| Climate | 700 |
| Pollution | 945 |
| Environment | 478 |
| Weather | 719 |
| Ozone Holes | 623 |
| Agriculture & Food | 844 |
| **Total** | **4,309** |

## 5. Data preprocessing
- Text = `Title` + `Body` (falls back to `Summary` when `Body` is empty; this happened for 1 record).
- Lower-casing, removal of URLs / e-mails / punctuation, whitespace normalisation (letters and digits are kept).
- `Date`, `Link`, `Year` and `Category` are not used as features.
- Empty-text removal (0 found) and duplicate-article removal (2 removed) -> **4,307** records used.

## 6. TF-IDF feature extraction
`TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000, min_df=3, sublinear_tf=True)` inside an sklearn `Pipeline`, so it is fitted only on training data (no leakage).

## 7. Train-test split
80% / 20%, `random_state=42`, stratified: **3,445 training** and **862 testing** records. All six categories appear in both sets.

## 8. Models trained
1. Multinomial Naive Bayes
2. Logistic Regression (`C=10`)
3. Support Vector Classifier: `LinearSVC` wrapped in `CalibratedClassifierCV` (sigmoid, 5-fold) to obtain genuine probabilities. (`SVC(probability=True)` is deprecated in scikit-learn 1.9 and took ~170 s per fit on the build machine.)

## 9. Model comparison (5-fold stratified CV on the training set)

| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Support Vector Classifier (SVC) | 0.7692 | 0.7586 | 0.7482 | **0.7493** | 0.7642 |
| 2 | Logistic Regression | 0.7646 | 0.7594 | 0.7408 | 0.7439 | 0.7591 |
| 3 | Multinomial Naive Bayes | 0.6412 | 0.5912 | 0.5824 | 0.5603 | 0.5994 |

The model with the highest mean CV Macro F1 was selected: **Support Vector Classifier (SVC)**. Its lead over Logistic Regression (0.0054) is smaller than its fold-to-fold standard deviation (0.0085), so the two are close; the lead over Naive Bayes is large.

## 10-11. Evaluation metrics and actual results (untouched test set, 862 articles)

| Metric | Value |
|---|---:|
| Accuracy | 0.7773 |
| Macro Precision | 0.7615 |
| Macro Recall | 0.7524 |
| Macro F1 | 0.7507 |
| Weighted F1 | 0.7691 |

| Category | Precision | Recall | F1 | Test articles |
|---|---:|---:|---:|---:|
| Climate | 0.6382 | 0.6929 | 0.6644 | 140 |
| Pollution | 0.7935 | 0.7725 | 0.7828 | 189 |
| Environment | 0.5965 | 0.3542 | 0.4444 | 96 |
| Weather | 0.7829 | 0.8264 | 0.8041 | 144 |
| Ozone Holes | 0.9426 | 0.9274 | 0.9350 | 124 |
| Agriculture & Food | 0.8154 | 0.9408 | 0.8736 | 169 |

Ozone Holes and Agriculture & Food are classified best. **Environment is the weakest class** (recall 0.35); its articles are most often predicted as Climate (22), Pollution (18) or Agriculture & Food (16). Categories overlap thematically, which limits accuracy.

The final label is the category with the highest predicted probability (on the test set this agreed with the classifier's own `.predict()` for 100% of articles).

## 12. Project folder structure
```
SciDCC_Climate_Article_Classifier/
├── data/            SciDCC.csv, filtered_SciDCC.csv
├── notebooks/       SciDCC_Climate_Change_Classification.ipynb
├── models/          best_model.joblib, best_model_name.txt, svc_model.joblib
├── reports/         dataset_statistics.json, model_comparison.csv,
│                    best_model_metrics.json, classification_report.csv
├── images/          category_distribution.png, wordcloud.png, model_comparison.png,
│                    confusion_matrix.png, article_length_distribution.png,
│                    original_category_distribution.png
├── app.py
├── requirements.txt, runtime.txt
├── README.md, LICENSE, .gitignore, .gitattributes
├── .streamlit/config.toml
└── .github/workflows/python-check.yml
```

## 13. Installation
Use Python 3.12 (the version used to build and test the models).
```bash
cd SciDCC_Climate_Article_Classifier
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
The saved `.joblib` models were created with scikit-learn 1.9.1; `requirements.txt` pins that version so they load without warnings.

## 14. Notebook execution
```bash
jupyter notebook notebooks/SciDCC_Climate_Change_Classification.ipynb
```
Choose *Kernel -> Restart & Run All*. The notebook finds `data/SciDCC.csv` automatically and rewrites the models, reports and images. Training takes roughly one to two minutes on a laptop.

Note: the `wordcloud` package was not installable where this project was built, so the committed `images/wordcloud.png` is a top-word frequency panel for each category. With `wordcloud` installed (it is in `requirements.txt`), re-running the notebook produces real word clouds.

## 15. Streamlit execution
```bash
streamlit run app.py
```

## 16. GitHub upload
```bash
git init
git add .
git commit -m "AI-Based Climate Change Article Classification System"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Largest tracked file is `data/SciDCC.csv` (about 44.6 MB), below GitHub's 100 MB hard limit (GitHub warns above 50 MB), so Git LFS is not required. The app itself only needs `models/` and `reports/`; if you want a smaller repository you may remove `data/` and keep only those folders.

## 17. Streamlit Community Cloud deployment
1. Push the repository to GitHub (section 16).
2. Open <https://share.streamlit.io>, sign in with GitHub and choose **Create app**.
3. Select the repository, branch `main` and main file `app.py`.
4. Open **Advanced settings** and choose **Python 3.12**.
5. Click **Deploy**.

About `runtime.txt`: Streamlit's documentation states that the Python version is chosen in the deployment dialog ("Advanced settings"), and community reports say Community Cloud ignores `runtime.txt`. The file is kept for other hosts that read it; do not rely on it for Community Cloud.
`requirements.txt` also lists Jupyter packages for local notebook work; for a leaner cloud build you may delete the `jupyter`, `notebook`, `ipykernel`, `seaborn` and `wordcloud` lines (the app does not import them).

## 18. Limitations
- Only six categories exist: the model **always** picks one, even for unrelated text, and a high probability is not proof of category membership.
- Overlapping topics (Climate / Weather / Environment / Pollution) cause many errors; Environment is especially weak.
- Trained on one news source (ScienceDaily) in one writing style; other sources are untested.
- Probabilities are calibrated on the training data but not independently verified; hyper-parameters were fixed, not tuned.
- Very short texts give unreliable predictions (the app rejects fewer than 10 words).

## 19. Future improvements
- Cross-validated hyper-parameter search; transformer-based models.
- An "Other / not relevant" class or out-of-domain detector.
- Reliability (calibration) analysis and testing on other sources.
- Show the words that most influenced a prediction.

## License
MIT - see `LICENSE` (replace `[YOUR NAME]` with your name).
