"""
Medicine Recommendation System — Prototype Web App
CS619 Final Year Project | Group ID: S26PROJECTFEF30

Run with:  streamlit run app.py
"""

import ast
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medicine Recommendation System",
    page_icon="💊",
    layout="wide",
)

DATA_DIR = "Datasets"
MODEL_DIR = "Models"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(f"{MODEL_DIR}/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(f"{MODEL_DIR}/symptom_columns.json") as f:
        symptom_columns = json.load(f)
    return model, le, symptom_columns


@st.cache_data
def load_lookup_tables():
    training = pd.read_csv(f"{DATA_DIR}/Training.csv")
    description = pd.read_csv(f"{DATA_DIR}/description.csv")
    precautions = pd.read_csv(f"{DATA_DIR}/precautions_df.csv")
    medications = pd.read_csv(f"{DATA_DIR}/medications.csv")
    diets = pd.read_csv(f"{DATA_DIR}/diets.csv")
    workout = pd.read_csv(f"{DATA_DIR}/workout_df.csv")
    severity = pd.read_csv(f"{DATA_DIR}/Symptom-severity.csv")
    return training, description, precautions, medications, diets, workout, severity


def normalize(name: str) -> str:
    """Collapse repeated whitespace and lowercase, for matching disease
    names across CSV files that were entered inconsistently."""
    return " ".join(str(name).split()).strip().lower()


# Manually verified alias for a genuine spelling inconsistency found in the
# source data ('diseae' -> 'disease') while building this prototype.
DISEASE_ALIASES = {
    "peptic ulcer diseae": "peptic ulcer disease",
}


def build_lookup_index(df, disease_col):
    index = {}
    for _, row in df.iterrows():
        key = normalize(row[disease_col])
        key = DISEASE_ALIASES.get(key, key)
        index[key] = row
    return index


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------
def get_recommendation(disease, description, precautions, medications, diets, workout):
    key = DISEASE_ALIASES.get(normalize(disease), normalize(disease))

    desc_idx = build_lookup_index(description, "Disease")
    prec_idx = build_lookup_index(precautions, "Disease")
    med_idx = build_lookup_index(medications, "Disease")
    diet_idx = build_lookup_index(diets, "Disease")

    desc_text = desc_idx[key]["Description"] if key in desc_idx else "No description available."

    prec_list = []
    if key in prec_idx:
        row = prec_idx[key]
        for c in [c for c in precautions.columns if "Precaution" in c]:
            if pd.notna(row[c]) and str(row[c]).strip():
                prec_list.append(str(row[c]))

    def parse_list_field(idx, field):
        if key not in idx:
            return []
        raw = idx[key][field]
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]

    med_list = parse_list_field(med_idx, "Medication")
    diet_list = parse_list_field(diet_idx, "Diet")

    workout_list = workout[workout["disease"].apply(lambda d: normalize(d) == key)]["workout"].tolist()

    return desc_text, prec_list, med_list, diet_list, workout_list


def predict_disease(model, le, symptom_columns, selected_symptoms):
    input_vec = np.zeros(len(symptom_columns))
    for s in selected_symptoms:
        if s in symptom_columns:
            input_vec[symptom_columns.index(s)] = 1
    input_df = pd.DataFrame([input_vec], columns=symptom_columns)
    pred_encoded = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    disease = le.inverse_transform([pred_encoded])[0]
    confidence = float(np.max(proba)) * 100
    # top-3 alternative predictions, for transparency
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], float(proba[i]) * 100) for i in top3_idx]
    return disease, confidence, top3


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
model, le, symptom_columns = load_model()
training, description, precautions, medications, diets, workout, severity = load_lookup_tables()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("💊 Medicine Recommendation System")
st.sidebar.caption("CS619 Prototype — Group S26PROJECTFEF30")
page = st.sidebar.radio(
    "Navigate",
    ["Symptom Checker", "Dataset Preview", "Model Evaluation"],
)

# ---------------------------------------------------------------------------
# Page: Symptom Checker
# ---------------------------------------------------------------------------
if page == "Symptom Checker":
    st.title("Symptom Checker & Recommendation")
    st.write(
        "Select the symptoms you are experiencing. The model will predict the "
        "most likely condition and suggest medicines, precautions, diet, and "
        "workout guidance."
    )

    display_names = sorted(s.replace("_", " ").title() for s in symptom_columns)
    display_to_raw = {s.replace("_", " ").title(): s for s in symptom_columns}

    selected_display = st.multiselect(
        "Select your symptoms",
        options=display_names,
        placeholder="Start typing a symptom, e.g. Headache, Fever...",
    )
    selected_symptoms = [display_to_raw[s] for s in selected_display]

    submitted = st.button("Get Recommendation", type="primary", disabled=len(selected_symptoms) == 0)

    if submitted:
        disease, confidence, top3 = predict_disease(model, le, symptom_columns, selected_symptoms)
        desc_text, prec_list, med_list, diet_list, workout_list = get_recommendation(
            disease, description, precautions, medications, diets, workout
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Predicted Condition: {disease}")
            st.write(desc_text)
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        with st.expander("Other possible matches"):
            for name, conf in top3:
                st.write(f"- {name}: {conf:.1f}%")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Suggested Medications**")
            for m in med_list:
                st.write(f"- {m}")
            st.markdown("**Precautions**")
            for p in prec_list:
                st.write(f"- {p}")
        with c2:
            st.markdown("**Suggested Diet**")
            for d in diet_list:
                st.write(f"- {d}")
            st.markdown("**Suggested Workout**")
            for w in workout_list:
                st.write(f"- {w}")

        st.info(
            "This prototype is for academic demonstration only and is not a "
            "substitute for professional medical advice.",
            icon="⚠️",
        )
    elif len(selected_symptoms) == 0:
        st.caption("Pick at least one symptom above, then click **Get Recommendation**.")

# ---------------------------------------------------------------------------
# Page: Dataset Preview
# ---------------------------------------------------------------------------
elif page == "Dataset Preview":
    st.title("Dataset Preview")
    st.write(f"Training set: **{training.shape[0]} rows × {training.shape[1]} columns** "
             f"({len(symptom_columns)} binary symptom features, {training['prognosis'].nunique()} diseases).")

    st.subheader("Sample rows")
    st.dataframe(training.head(10))

    st.subheader("Diseases covered")
    counts = training["prognosis"].value_counts().reset_index()
    counts.columns = ["Disease", "Rows"]
    st.dataframe(counts, height=300)

    st.subheader("Reference tables loaded")
    ref_info = pd.DataFrame(
        {
            "Table": ["description.csv", "precautions_df.csv", "medications.csv", "diets.csv", "workout_df.csv", "Symptom-severity.csv"],
            "Rows": [len(description), len(precautions), len(medications), len(diets), len(workout), len(severity)],
            "Purpose": [
                "Disease description shown in results",
                "Precautions to display per disease",
                "Recommended medications per disease",
                "Recommended diet per disease",
                "Recommended workout/lifestyle tips per disease",
                "Symptom severity weighting (reference)",
            ],
        }
    )
    st.dataframe(ref_info, hide_index=True)

# ---------------------------------------------------------------------------
# Page: Model Evaluation
# ---------------------------------------------------------------------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write(
        "Metrics below are computed from the held-out 20% test split created "
        "during training (see `Notebooks/training.ipynb`)."
    )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = training[symptom_columns]
    y_true_raw = training["prognosis"]
    y_true = le.transform(y_true_raw)
    _, X_test, _, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42, stratify=y_true)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Test Accuracy", f"{acc * 100:.2f}%")
    st.caption(
        "Note: this dataset encodes each disease as a fixed combination of "
        "symptoms, so the classes are cleanly separable and near-perfect "
        "accuracy is expected — this is a known property of this dataset, "
        "not a sign of data leakage."
    )

    st.subheader("Model details")
    st.write(f"Algorithm: **Random Forest Classifier** (200 trees)")
    st.write(f"Features: {len(symptom_columns)} binary symptom indicators")
    st.write(f"Classes: {len(le.classes_)} diseases")

    st.subheader("Feature importance (top 15 symptoms)")
    importances = pd.Series(model.feature_importances_, index=symptom_columns)
    top15 = importances.sort_values(ascending=False).head(15)
    st.bar_chart(top15)
