"""
app/cardio_tab.py — Cardiology tab
===================================
Heart disease risk assessment: takes the 13 clinical features, predicts with
Random Forest and DNN, and explains each prediction with SHAP waterfall plots
and LIME local explanations.

Imported by app.py, which is the Streamlit entry point. This module only draws,
it does not run anything on import. The two model loaders are cached with
@st.cache_resource, so the models are read from disk once per session.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, must precede the pyplot import
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap
import lime
import lime.lime_tabular


THESIS_DIR    = Path(__file__).parent.parent.resolve()
RF_MODEL_PATH = THESIS_DIR / "saved_models" / "rf_model.pkl"
DNN_MODEL_PATH= THESIS_DIR / "saved_models" / "dnn_model.keras"
X_TRAIN_PATH  = THESIS_DIR / "data" / "HDD" / "X_train_ready.csv"
SCALER_PATH   = THESIS_DIR / "data" / "HDD" / "scaler.pkl"

# Import mappings
if str(THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(THESIS_DIR))

from mappings import (
    sex_mapping, cp_mapping, fbs_mapping,
    restecg_mapping, exang_mapping, slope_mapping, thal_mapping,
)

# Feature metadata
FEATURE_NAMES_EN = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
    "Exercise Angina", "ST Depression", "ST Slope", "Major Vessels", "Thalassemia",
]


# Cached resource loaders

@st.cache_resource(show_spinner="Φόρτωση μοντέλων Καρδιολογίας / Loading cardiology models…")
def load_cardiology_resources():
    """Load RF, DNN, training data, and scaler once per session."""
    import tensorflow as tf  # local import — avoids loading TF for derm-only sessions

    rf     = joblib.load(str(RF_MODEL_PATH))
    dnn    = tf.keras.models.load_model(str(DNN_MODEL_PATH))
    X_train= pd.read_csv(str(X_TRAIN_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    return rf, dnn, X_train, scaler


@st.cache_resource(show_spinner="Προετοιμασία LIME / Preparing LIME explainer…")
def load_lime_explainer(_X_train: pd.DataFrame):
    """
    Build and cache the LIME TabularExplainer.
    Underscore prefix on _X_train tells Streamlit to skip hashing the DataFrame.
    The same explainer object is reused for both RF and DNN.
    """
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=_X_train.values,
        feature_names=FEATURE_NAMES_EN,
        class_names=["Healthy / Υγιής", "Heart Disease / Καρδιοπάθεια"],
        mode="classification",
        random_state=42,
    )


#  Cardiology helpers - SHAP
def _get_shap_class1(shap_vals, expected_val):
    """
    Normalize SHAP output to (values_1d, base_value) for class 1.
    Handles list / 2-D / 3-D array formats across shap versions.
    """
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1]).squeeze()
        bv = (expected_val[1] if isinstance(expected_val, (list, np.ndarray))
              else float(expected_val))
    elif np.array(shap_vals).ndim == 3:
        sv = np.array(shap_vals)[0, :, 1]
        bv = (expected_val[1] if isinstance(expected_val, (list, np.ndarray))
              else float(expected_val))
    else:
        sv = np.array(shap_vals).squeeze()
        bv = (float(expected_val[1]) if isinstance(expected_val, (list, np.ndarray))
              else float(expected_val))

    if sv.ndim > 1:
        sv = sv[0]
    return sv, bv


def compute_rf_shap(rf_model, X_scaled: np.ndarray):
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_scaled)
    return _get_shap_class1(shap_vals, explainer.expected_value)


def compute_dnn_shap(dnn_model, X_train: pd.DataFrame, X_scaled: np.ndarray):
    background = X_train.values[:100]
    base_value = float(dnn_model.predict(background, verbose=0).flatten().mean())

    # Attempt 1: DeepExplainer
    try:
        exp = shap.DeepExplainer(dnn_model, background)
        sv  = exp.shap_values(X_scaled)
        sv  = np.array(sv[0] if isinstance(sv, list) else sv).squeeze()
        if sv.ndim > 1:
            sv = sv[0]
        return sv, base_value
    except Exception:
        pass

    # Attempt 2: GradientExplainer
    try:
        exp = shap.GradientExplainer(dnn_model, background)
        sv  = exp.shap_values(X_scaled)
        sv  = np.array(sv[0] if isinstance(sv, list) else sv).squeeze()
        if sv.ndim > 1:
            sv = sv[0]
        return sv, base_value
    except Exception:
        pass

    # Attempt 3: KernelExplainer (slow but universal)
    exp = shap.KernelExplainer(
        lambda x: dnn_model.predict(x, verbose=0).flatten(),
        background[:30],
    )
    sv  = exp.shap_values(X_scaled, nsamples=100)
    sv  = np.array(sv).squeeze()
    bv  = float(exp.expected_value)
    return sv, bv


def make_waterfall_fig(sv: np.ndarray, bv: float, raw_row: np.ndarray) -> plt.Figure:
    """Build a SHAP waterfall matplotlib Figure."""
    explanation = shap.Explanation(
        values      = sv,
        base_values = bv,
        data        = np.round(raw_row, 2),
        feature_names = FEATURE_NAMES_EN,
    )
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    fig = plt.gcf()
    return fig


#  Cardiology helpers - LIME
def make_lime_fig(
    explainer: lime.lime_tabular.LimeTabularExplainer,
    predict_fn,
    X_scaled_1d: np.ndarray,
    title: str,
) -> plt.Figure:
    """Run LIME explain_instance and return its matplotlib Figure."""
    exp = explainer.explain_instance(
        data_row  = X_scaled_1d,
        predict_fn= predict_fn,
        num_features = 10,
        num_samples  = 300,   # faster than default 5000; good enough for 1 patient
    )
    fig = exp.as_pyplot_figure()
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def cardio_consensus(rf_prob: float, dnn_prob: float) -> None:
    agree = (rf_prob > 0.5) == (dnn_prob > 0.5)
    if agree:
        label = "Καρδιοπάθεια / Heart Disease" if rf_prob > 0.5 else "Υγιής / Healthy"
        st.success(f"**Σύγκλιση / Consensus** — RF & DNN συμφωνούν: **{label}**")
    else:
        st.warning(
            "**Διαφωνία / Disagreement** — RF & DNN διαφωνούν.\n\n"
            "Συνιστάται κλινικός έλεγχος / Clinical review recommended."
        )


def render_cardiology_tab() -> None:
    st.header("Καρδιολογική Εκτίμηση Κινδύνου / Cardiology Risk Assessment")
    # st.markdown(
    #     "Εισάγετε τα κλινικά δεδομένα του ασθενή (Cleveland Heart Disease Dataset). "
    #     "Το σύστημα εκτιμά τον κίνδυνο καρδιοπάθειας με **Random Forest** και **DNN**, "
    #     "και παράγει επεξηγήσεις με **SHAP** και **LIME**.\n\n"
    #     "*Enter patient clinical data. The system estimates heart disease risk with RF and DNN, "
    #     "and explains decisions using SHAP and LIME.*"
    # )
    # st.divider()

    # Load resources (cached after first call)
    rf_model, dnn_model, X_train, scaler = load_cardiology_resources()
    lime_explainer = load_lime_explainer(X_train)

    # Input form
    st.subheader("Κλινικά Δεδομένα Ασθενή / Patient Clinical Data")

    with st.form("cardiology_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Γενικά Στοιχεία & Παράγοντες Κινδύνου / General Info & Risk Factors**")
            age  = st.number_input("Ηλικία / Age (έτη/years)", 20, 100, 55)
            sex  = st.selectbox(
                "Φύλο / Sex",
                options=list(sex_mapping.keys()),
                format_func=lambda k: sex_mapping[k],
            )
            chol = st.number_input("Χοληστερόλη Ορού / Serum Cholesterol (mg/dl)", 100, 600, 250)
            fbs  = st.selectbox(
                "Σάκχαρο Νηστείας / Fasting Blood Sugar",
                options=list(fbs_mapping.keys()),
                format_func=lambda k: fbs_mapping[k],
            )
            thal = st.selectbox(
                "Θαλασσαιμία / Thalassemia",
                options=list(thal_mapping.keys()),
                format_func=lambda k: thal_mapping[k],
            )

        with c2:
            st.markdown("**Κλινική Εξέταση & Συμπτώματα / Clinical Examination & Symptoms**")
            cp       = st.selectbox(
                "Τύπος Θωρακικού Άλγους / Chest Pain Type",
                options=list(cp_mapping.keys()),
                format_func=lambda k: cp_mapping[k],
            )
            trestbps = st.number_input("Αρτηριακή Πίεση Ηρεμίας / Resting BP (mm Hg)", 80, 220, 130)
            thalach  = st.number_input("Μέγιστη Καρδιακή Συχνότητα / Max Heart Rate (bpm)", 60, 220, 150)
            exang    = st.selectbox(
                "Στηθάγχη Άσκησης / Exercise Induced Angina",
                options=list(exang_mapping.keys()),
                format_func=lambda k: exang_mapping[k],
            )

        with c3:
            st.markdown("**Ηλεκτροκαρδιογραφικά Ευρήματα / ECG & Angiography Findings**")
            restecg = st.selectbox(
                "ΗΚΓ Ηρεμίας / Resting ECG",
                options=list(restecg_mapping.keys()),
                format_func=lambda k: restecg_mapping[k],
            )
            oldpeak = st.number_input(
                "ST Κατάσπαση / ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1,
            )
            slope   = st.selectbox(
                "Κλίση ST / Slope of ST Segment",
                options=list(slope_mapping.keys()),
                format_func=lambda k: slope_mapping[k],
            )
            ca      = st.selectbox(
                "Κύρια Αγγεία (φθοριοσκοπία) / Major Vessels (fluoroscopy)",
                options=[0, 1, 2, 3],
            )

        submitted = st.form_submit_button(
            "Εκτίμηση Κινδύνου / Assess Risk",
            type="primary",
            width='stretch',
        )

    if not submitted:
        return

    # Preprocess
    X_raw    = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]],
                        dtype=np.float64)
    X_scaled = scaler.transform(X_raw)

    # Predictions
    rf_prob  = float(rf_model.predict_proba(X_scaled)[0, 1])
    dnn_prob = float(dnn_model.predict(X_scaled, verbose=0)[0, 0])

    st.divider()
    st.subheader("Αποτελέσματα Πρόβλεψης / Prediction Results")

    pm1, pm2, pm3 = st.columns(3)
    with pm1:
        st.metric(
            label="Random Forest",
            value="🔴 Καρδιοπάθεια / Heart Disease" if rf_prob > 0.5 else "🟢 Υγιής / Healthy",
            delta=f"P(disease) = {rf_prob * 100:.1f}%",
        )
    with pm2:
        st.metric(
            label="Deep Neural Network",
            value="🔴 Καρδιοπάθεια / Heart Disease" if dnn_prob > 0.5 else "🟢 Υγιής / Healthy",
            delta=f"P(disease) = {dnn_prob * 100:.1f}%",
        )
    with pm3:
        avg = (rf_prob + dnn_prob) / 2.0
        st.metric(
            label="Ensemble — Μέσος Όρος / Average",
            value="🔴 Καρδιοπάθεια / Heart Disease" if avg > 0.5 else "🟢 Υγιής / Healthy",
            delta=f"P(disease) = {avg * 100:.1f}%",
        )

    cardio_consensus(rf_prob, dnn_prob)

    # SHAP
    st.divider()
    st.subheader("SHAP — Waterfall Plots")
    st.caption(
        "Κάθε μπάρα δείχνει πόσο κάθε χαρακτηριστικό ωθεί την πρόβλεψη "
        "προς Καρδιοπάθεια (+) ή Υγιής (−). "
        "/ Each bar shows how much each feature pushes the prediction toward Heart Disease (+) or Healthy (−)."
    )

    shap_c1, shap_c2 = st.columns(2)

    with shap_c1:
        st.markdown("**Random Forest**")
        with st.spinner("Υπολογισμός RF SHAP… / Computing RF SHAP…"):
            rf_sv, rf_bv   = compute_rf_shap(rf_model, X_scaled)
            fig_rf_shap    = make_waterfall_fig(rf_sv, rf_bv, X_raw[0])
            st.pyplot(fig_rf_shap, width='stretch')
            plt.close(fig_rf_shap)

    with shap_c2:
        st.markdown("**Deep Neural Network**")
        with st.spinner("Υπολογισμός DNN SHAP… / Computing DNN SHAP… (~10–30 sec)"):
            dnn_sv, dnn_bv = compute_dnn_shap(dnn_model, X_train, X_scaled)
            fig_dnn_shap   = make_waterfall_fig(dnn_sv, dnn_bv, X_raw[0])
            st.pyplot(fig_dnn_shap, width='stretch')
            plt.close(fig_dnn_shap)

    # with st.expander("ℹ️ Επεξήγηση SHAP / About SHAP"):
    #     st.markdown(
    #         "**SHAP** (SHapley Additive exPlanations): Μοιράζει τη διαφορά μεταξύ της πρόβλεψης "
    #         "και της μέσης πρόβλεψης σε όλα τα χαρακτηριστικά, βάσει θεωρίας παιγνίων.\n\n"
    #         "*Distributes the difference between the prediction and the average prediction "
    #         "across all features, based on game theory (Shapley values).*"
    #     )

    # LIME
    st.divider()
    st.subheader("LIME — Local Explanations")
    st.caption(
        "Το LIME προσαρμόζει ένα τοπικό γραμμικό μοντέλο γύρω από τον συγκεκριμένο ασθενή "
        "για να εξηγήσει αυτή την πρόβλεψη. "
        "/ LIME fits a local linear model around this specific patient to explain the prediction."
    )

    def rf_predict_fn(data):
        return rf_model.predict_proba(data)

    def dnn_predict_fn(data):
        preds = dnn_model.predict(data, verbose=0).flatten()
        return np.column_stack([1.0 - preds, preds])

    lime_c1, lime_c2 = st.columns(2)

    with lime_c1:
        st.markdown("**Random Forest**")
        with st.spinner("Υπολογισμός RF LIME… / Computing RF LIME…"):
            fig_rf_lime = make_lime_fig(
                lime_explainer, rf_predict_fn,
                X_scaled[0], "Random Forest — LIME",
            )
            st.pyplot(fig_rf_lime, width='stretch')
            plt.close(fig_rf_lime)

    with lime_c2:
        st.markdown("**Deep Neural Network**")
        with st.spinner("Υπολογισμός DNN LIME… / Computing DNN LIME…"):
            fig_dnn_lime = make_lime_fig(
                lime_explainer, dnn_predict_fn,
                X_scaled[0], "DNN — LIME",
            )
            st.pyplot(fig_dnn_lime, width='stretch')
            plt.close(fig_dnn_lime)

    # with st.expander("Επεξήγηση LIME / About LIME"):
    #     st.markdown(
    #         "**LIME** (Local Interpretable Model-Agnostic Explanations): Δημιουργεί τεχνητά δείγματα "
    #         "γύρω από τον ασθενή, τα ταξινομεί, και προσαρμόζει ένα απλό γραμμικό μοντέλο.\n\n"
    #         "*Creates artificial samples around the patient, classifies them, and fits a simple "
    #         "linear model to approximate local decision boundaries.*"
    #     )
