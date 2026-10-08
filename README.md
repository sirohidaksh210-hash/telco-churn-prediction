# 📉 Customer Churn Prediction: EDA, Modeling & Live App

End-to-end machine-learning project that predicts whether a telecom customer will churn.

| | |
|---|---|
| **Task 1** | EDA + preprocessing + model comparison notebook: [`Churn_EDA_Modeling.ipynb`](Churn_EDA_Modeling.ipynb) |
| **Task 2** | Interactive Streamlit prediction dashboard: [`app.py`](app.py) |
| **Live demo** | 👉 `<ADD YOUR STREAMLIT / HUGGING FACE LINK HERE>` |
| **Colab / Kaggle notebook** | `<ADD YOUR NOTEBOOK LINK HERE (read access enabled)>` |

## Dataset
IBM **Telco Customer Churn**: 7,043 customers, 21 columns (demographics, subscribed services, account/billing info, `Churn` label). ~26.5% of customers churned (imbalanced).
Source: <https://github.com/IBM/telco-customer-churn-on-icp4d> (a copy is stored in `data/`).

## Approach
- **Cleaning:** blank `TotalCharges` (11 new customers) → median imputation *inside* the pipeline; `customerID` dropped.
- **EDA:** target balance, univariate (histograms, box plots, count plots), bivariate (features vs. churn), correlation heatmap.
- **Preprocessing (leak-free):** stratified 80/20 split first; median imputation, IQR winsorizing, standard scaling and one-hot encoding are fitted on training data only, inside a scikit-learn `Pipeline`.
- **Feature engineering:** `NumServices` (count of subscribed services). A `TotalCharges/tenure` feature was tested and dropped (≈1.00 correlation with `MonthlyCharges`).
- **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting; 5-fold stratified CV, `GridSearchCV` on the two ensemble models.
- **Threshold tuning:** decision threshold chosen on out-of-fold *training* predictions (0.34) to improve recall on churners.

## Model performance (held-out test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting (tuned), deployed** | 0.800 | 0.659 | 0.511 | 0.575 | **0.845** |
| Random Forest (tuned) | 0.770 | 0.547 | 0.773 | 0.641 | 0.843 |
| Logistic Regression | 0.739 | 0.505 | 0.783 | 0.614 | 0.841 |
| Decision Tree | 0.755 | 0.527 | 0.759 | 0.622 | 0.833 |

Deployed model at the tuned threshold of 0.34: **Accuracy 0.778 · Precision 0.563 · Recall 0.725 · F1 0.634**.
Top drivers: month-to-month contract, short tenure, fiber-optic internet, no online security/tech support, electronic-check payment.

## App features
- Form with 19 inputs, dependent fields auto-disabled (e.g. no internet → add-ons locked), input validation and clear error messages
- Real-time churn probability, risk band (Low / Medium / High), probability bar with decision threshold, retention suggestions
- Batch scoring via CSV upload with file-type / missing-column / bad-value handling and downloadable results
- Model performance tab

## Project structure
```
├── app.py                       # Streamlit app
├── churn_utils.py               # feature engineering + Winsorizer (shared by notebook & app)
├── Churn_EDA_Modeling.ipynb     # Task 1 notebook (executed, with outputs)
├── data/Telco-Customer-Churn.csv
├── models/churn_model.joblib    # full preprocessing + model pipeline
├── models/model_card.joblib     # metrics + decision threshold
└── requirements.txt
```

## Tech stack
Python 3.11+, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit.

## Run locally
```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                    # use the pre-trained model
```
**Reproduce training:** `jupyter notebook Churn_EDA_Modeling.ipynb` → *Run all*. This regenerates `models/*.joblib`.

> ⚠️ `scikit-learn` is pinned to **1.8.0** because the saved model is a pickle; loading it with a different version may fail. If you retrain with another version, the pin can be changed.

## Deployment (Streamlit Community Cloud)
1. Push this repo to a public GitHub repository.
2. Go to <https://share.streamlit.io> → **New app** → select the repo, branch `main`, main file `app.py`.
3. In *Advanced settings* choose Python 3.11 or newer, then deploy and paste the URL above.
