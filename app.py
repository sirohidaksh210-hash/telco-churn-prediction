"""Streamlit app: real-time customer churn risk prediction."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_utils import RAW_FEATURES, add_features

BASE = Path(__file__).parent
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "models" / "churn_model.joblib")
    card = joblib.load(BASE / "models" / "model_card.joblib")
    return model, card


try:
    model, card = load_artifacts()
except Exception as e:  # missing / incompatible model file
    st.error(f"Could not load the trained model: {e}. Run the notebook first to create `models/churn_model.joblib`.")
    st.stop()

THRESHOLD = card["threshold"]


def predict(df_raw: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(add_features(df_raw)[RAW_FEATURES + ["NumServices"]])[:, 1], index=df_raw.index)


def risk_band(p: float):
    if p >= 0.6:
        return "High", "#d62728"
    if p >= THRESHOLD:
        return "Medium", "#ff7f0e"
    return "Low", "#2ca02c"


def prob_bar(p: float, color: str):
    st.markdown(
        f"""<div style="background:#e6e6e6;border-radius:10px;height:26px;position:relative;">
        <div style="width:{p*100:.1f}%;background:{color};height:26px;border-radius:10px;"></div>
        <div style="position:absolute;left:{THRESHOLD*100:.0f}%;top:-4px;height:34px;border-left:2px dashed #333;"></div></div>
        <div style="font-size:12px;color:#666;margin-top:4px;">Dashed line = decision threshold ({THRESHOLD:.2f}). Customers above it are flagged as likely churners.</div>""",
        unsafe_allow_html=True)


st.title("📉 Customer Churn Predictor")
st.caption(f"Model: **{card['model_name']}**, trained on the IBM Telco Customer Churn dataset (7,043 customers).")

with st.expander("ℹ️ How to use this tool", expanded=False):
    st.markdown(
        "1. **Single customer** tab: fill in the customer's details and press **Predict churn risk**.\n"
        "2. **Batch (CSV)** tab: upload a CSV with the same columns as the Telco dataset to score many customers.\n"
        "3. The result is the model's estimated **probability of churn**, plus a risk band (Low / Medium / High).\n\n"
        "This is a decision-support estimate from a statistical model, not a certainty.")

tab1, tab2, tab3 = st.tabs(["Single customer", "Batch (CSV)", "Model performance"])

# ---------------- Single prediction ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", 0, 72, 12, help="Months the customer has been with the company (0-72).")
    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes"], disabled=(phone == "No"))
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        has_net = internet != "No"
        addons = {}
        for label, key in [("Online security", "OnlineSecurity"), ("Online backup", "OnlineBackup"),
                           ("Device protection", "DeviceProtection"), ("Tech support", "TechSupport"),
                           ("Streaming TV", "StreamingTV"), ("Streaming movies", "StreamingMovies")]:
            addons[key] = st.selectbox(label, ["No", "Yes"], disabled=not has_net, key=key)
    with c3:
        st.subheader("Account & billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
        know_total = st.checkbox("I know the total charges")
        total = st.number_input("Total charges ($)", 0.0, 20000.0, float(tenure * monthly), step=1.0) if know_total else None

    if st.button("Predict churn risk", type="primary", width="stretch"):
        # ---- validation ----
        problems = []
        if monthly <= 0 and (has_net or phone == "Yes"):
            problems.append("Monthly charges must be greater than 0 for a customer with active services.")
        if not has_net and not (phone == "Yes"):
            problems.append("A customer needs at least phone or internet service.")
        if know_total and tenure > 0 and total is not None and total < monthly * 0.5:
            problems.append("Total charges look too low for this tenure and monthly charge. Please check the values.")
        if problems:
            for p_ in problems:
                st.error(p_)
        else:
            row = {"gender": gender, "SeniorCitizen": "1" if senior == "Yes" else "0", "Partner": partner,
                   "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
                   "MultipleLines": "No phone service" if phone == "No" else multi, "InternetService": internet,
                   **{k: ("No internet service" if not has_net else v) for k, v in addons.items()},
                   "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
                   "MonthlyCharges": monthly, "TotalCharges": total if know_total else tenure * monthly}
            # SeniorCitizen was cast to str in training (add_features); keep 0/1 numeric in raw frame
            row["SeniorCitizen"] = int(row["SeniorCitizen"])
            try:
                p = float(predict(pd.DataFrame([row])).iloc[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            else:
                band, color = risk_band(p)
                st.divider()
                m1, m2 = st.columns([1, 2])
                m1.metric("Churn probability", f"{p:.1%}")
                m1.markdown(f"<h3 style='color:{color};margin:0'>{band} risk</h3>", unsafe_allow_html=True)
                with m2:
                    prob_bar(p, color)
                if p >= THRESHOLD:
                    tips = []
                    if contract == "Month-to-month":
                        tips.append("offer a discount to move to a 1- or 2-year contract")
                    if has_net and addons["OnlineSecurity"] == "No" and addons["TechSupport"] == "No":
                        tips.append("bundle online security / tech support")
                    if payment == "Electronic check":
                        tips.append("encourage automatic payment")
                    if tips:
                        st.info("**Retention ideas:** " + "; ".join(tips) + ".")
                else:
                    st.success("This customer looks relatively stable.")

# ---------------- Batch ----------------
with tab2:
    st.write("Upload a **CSV** containing these columns: " + ", ".join(f"`{c}`" for c in RAW_FEATURES))
    up = st.file_uploader("CSV file", type=None)
    if up is not None:
        if not up.name.lower().endswith(".csv"):
            st.error("Unsupported file type. Please upload a .csv file.")
        else:
            try:
                data = pd.read_csv(up)
            except Exception:
                st.error("Could not read this file as CSV.")
                data = None
            if data is not None:
                missing = [c for c in RAW_FEATURES if c not in data.columns]
                if data.empty:
                    st.error("The file has no rows.")
                elif missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                else:
                    try:
                        data = data.copy()
                        data["Churn probability"] = predict(data[RAW_FEATURES]).round(4)
                        data["Flagged as likely churn"] = data["Churn probability"] >= THRESHOLD
                    except Exception as e:
                        st.error(f"Could not score this file (check for unexpected category values): {e}")
                    else:
                        st.success(f"Scored {len(data):,} customers. {int(data['Flagged as likely churn'].sum()):,} flagged.")
                        st.dataframe(data.sort_values("Churn probability", ascending=False), width="stretch")
                        st.download_button("Download results", data.to_csv(index=False), "churn_predictions.csv", "text/csv")

# ---------------- Performance ----------------
with tab3:
    st.write(f"Metrics on a held-out test set of 1,409 customers. The deployed model uses a tuned decision threshold of **{THRESHOLD:.2f}**.")
    mt = card["metrics_at_threshold"]
    cols = st.columns(4)
    for col, k in zip(cols, ["Accuracy", "Precision", "Recall", "F1"]):
        col.metric(k, f"{mt[k]:.3f}")
    st.metric("ROC-AUC", f"{card['metrics']['ROC-AUC']:.3f}")
    st.markdown("**All models compared (default 0.5 threshold):**")
    st.dataframe(pd.DataFrame(card["all_metrics"]).set_index("Model").round(3), width="stretch")
