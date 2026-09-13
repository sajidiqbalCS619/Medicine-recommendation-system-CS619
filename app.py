"""
Medicine Recommendation System — AI Chatbot Assistant
CS619 Final Year Project | Group ID: S26PROJECTFEF30

Run with:  streamlit run app.py
"""

import ast
import json
import pickle
import re
import difflib

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medicine Recommendation Assistant",
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
# source data ('diseae' -> 'disease').
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
# Recommendation logic (Phase 4)
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
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], float(proba[i]) * 100) for i in top3_idx]
    return disease, confidence, top3


# ---------------------------------------------------------------------------
# NLP symptom extraction from free text (Phase 4 — the chatbot's "ears")
# ---------------------------------------------------------------------------
# A modest, manually-curated synonym map for common everyday phrasing that
# won't fuzzy-match closely enough to the clinical column names on their own.
SYNONYMS = {
    "throwing up": "vomiting", "puking": "vomiting", "nauseous": "nausea",
    "feel sick": "nausea", "tummy ache": "stomach_pain", "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain", "high temperature": "high_fever", "temperature": "mild_fever",
    "cant sleep": "insomnia", "can not sleep": "insomnia", "trouble sleeping": "insomnia",
    "runny nose": "continuous_sneezing", "sore throat": "patches_in_throat", "tired": "fatigue",
    "exhausted": "fatigue", "dizzy": "dizziness", "out of breath": "breathlessness",
    "short of breath": "breathlessness", "yellow skin": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes", "itchy skin": "itching", "itchy": "itching",
    "rash": "skin_rash",
}

STOPWORDS = {
    "i", "have", "has", "had", "am", "is", "are", "my", "me", "a", "an", "the",
    "and", "or", "feeling", "feel", "been", "im", "ive", "also", "with", "some",
    "lot", "of", "really", "very", "bit", "little", "since", "for", "days", "day",
}


def _clean_phrase(s):
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip().lower()


def _tokenize(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return [w for w in text.split() if w not in STOPWORDS]


def _word_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_symptoms(text, symptom_columns, fuzzy_threshold=0.85):
    """Free-text symptom extraction: exact multi-word phrase match first,
    then a length-matched fuzzy fallback, then single-word matches. Returns
    a dict of {symptom_key: matched surface text}."""
    phrase_bank = {c: _clean_phrase(c) for c in symptom_columns}
    text_norm = re.sub(r"\s+", " ", text.lower().strip())
    matched = {}
    words = _tokenize(text)
    used_idx = set()

    # 1. Curated synonyms — highest confidence
    for phrase, target in SYNONYMS.items():
        if phrase in text_norm and target in phrase_bank and target not in matched:
            matched[target] = phrase

    remaining = [(k, p) for k, p in phrase_bank.items() if k not in matched]
    remaining.sort(key=lambda kp: len(kp[1].split()), reverse=True)

    # 2. Exact multi-word phrase containment
    for key, phrase in remaining:
        phrase_words = phrase.split()
        plen = len(phrase_words)
        if plen < 2:
            continue
        for i in range(len(words) - plen + 1):
            if any(j in used_idx for j in range(i, i + plen)):
                continue
            if words[i:i + plen] == phrase_words:
                matched[key] = " ".join(words[i:i + plen])
                used_idx.update(range(i, i + plen))
                break

    # 3. Length-matched fuzzy fallback (word-level, not character-level, so
    #    unrelated same-length phrases don't get confused with each other)
    remaining = [(k, p) for k, p in remaining if k not in matched]
    for key, phrase in remaining:
        phrase_words = phrase.split()
        plen = len(phrase_words)
        if plen < 2:
            continue
        best_i, best_score = None, 0.0
        for i in range(len(words) - plen + 1):
            if any(j in used_idx for j in range(i, i + plen)):
                continue
            window = words[i:i + plen]
            hits = sum(1 for a, b in zip(window, phrase_words) if _word_ratio(a, b) >= fuzzy_threshold)
            score = hits / plen
            if score > best_score:
                best_score, best_i = score, i
        if best_score >= 0.67:
            matched[key] = " ".join(words[best_i:best_i + plen])
            used_idx.update(range(best_i, best_i + plen))

    # 4. Single-word symptoms — exact, then close-typo fuzzy match
    single_word = [(k, p) for k, p in phrase_bank.items() if k not in matched and " " not in p]
    for i, w in enumerate(words):
        if i in used_idx:
            continue
        for key, phrase in single_word:
            if key in matched:
                continue
            if w == phrase or _word_ratio(w, phrase) >= fuzzy_threshold:
                matched[key] = w
                used_idx.add(i)
                break

    return matched


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
model, le, symptom_columns = load_model()
training, description, precautions, medications, diets, workout, severity = load_lookup_tables()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("💊 Medicine Recommendation Assistant")
st.sidebar.caption("CS619 Prototype — Group S26PROJECTFEF30")
page = st.sidebar.radio(
    "Navigate",
    ["AI Chatbot", "Manual Symptom Checker", "Dataset Preview", "Model Evaluation"],
)

# ---------------------------------------------------------------------------
# Page: AI Chatbot (primary interface)
# ---------------------------------------------------------------------------
if page == "AI Chatbot":
    st.title("AI Medicine Recommendation Assistant")
    st.caption(
        "Describe how you're feeling in your own words — e.g. \"I have itching, "
        "a skin rash, and some nodal skin eruptions.\" The assistant will pick out "
        "the symptoms it recognizes and suggest what might be going on."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Tell me what symptoms you're experiencing, in your own words, "
                    "and I'll try to help figure out what might be going on."
                ),
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Describe your symptoms...")

    if user_text:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        detected = extract_symptoms(user_text, symptom_columns)

        with st.chat_message("assistant"):
            if not detected:
                reply = (
                    "I couldn't confidently pick out any symptoms from that — could you "
                    "try describing them more specifically? For example: \"I have "
                    "itching, joint pain, and a high temperature.\""
                )
                st.markdown(reply)
            else:
                detected_names = [k.replace("_", " ").title() for k in detected.keys()]
                disease, confidence, top3 = predict_disease(model, le, symptom_columns, list(detected.keys()))
                desc_text, prec_list, med_list, diet_list, workout_list = get_recommendation(
                    disease, description, precautions, medications, diets, workout
                )

                lines = []
                lines.append(f"**Detected symptoms:** {', '.join(detected_names)}")
                lines.append(f"\n### Predicted Condition: {disease}")
                lines.append(f"**Confidence:** {confidence:.1f}%")
                lines.append(f"\n{desc_text}")

                lines.append("\n**Suggested Medications**")
                for m in med_list:
                    lines.append(f"- {m}")

                lines.append("\n**Precautions**")
                for p in prec_list:
                    lines.append(f"- {p}")

                lines.append("\n**Suggested Diet**")
                for d in diet_list:
                    lines.append(f"- {d}")

                lines.append("\n**Suggested Workout**")
                for w in workout_list:
                    lines.append(f"- {w}")

                if len(detected) == 1:
                    lines.append(
                        "\n*I only picked up one symptom — feel free to describe more "
                        "for a more confident prediction.*"
                    )

                lines.append(
                    "\n---\n*This is an academic prototype, not a substitute for "
                    "professional medical advice.*"
                )
                reply = "\n".join(lines)
                st.markdown(reply)

                with st.expander("Other possible matches"):
                    for name, conf in top3:
                        st.write(f"- {name}: {conf:.1f}%")

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.get("chat_history") and len(st.session_state.chat_history) > 1:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Page: Manual Symptom Checker (kept for users who prefer direct selection)
# ---------------------------------------------------------------------------
elif page == "Manual Symptom Checker":
    st.title("Manual Symptom Checker")
    st.write(
        "Prefer picking symptoms from a list instead of describing them? Use this "
        "instead of the chatbot."
    )

    display_names = sorted(s.replace("_", " ").title() for s in symptom_columns)
    display_to_raw = {s.replace("_", " ").title(): s for s in symptom_columns}

    selected_display = st.multiselect("Select your symptoms", options=display_names)
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
