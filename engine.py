"""
engine.py - Recommendation Engine (Design Document: application tier).

Covers FR-17 .. FR-24 and UC-03:
  free-text -> symptoms (NLP) -> optional profile (age, gender, allergy, pregnancy)
  -> disease prediction by the ACTIVE model -> ranked medicines + care guidance.

The NLP symptom extractor is the one accepted in the prototype (app.py v1).
The chatbot stays free-text only: age group, gender and allergies are read from
the sentence itself ("I am 30, female, allergic to penicillin, with fever...").
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import db
import reference_data as ref
import trainer

LOW_CONFIDENCE_THRESHOLD = 50.0  # SRS UC-03 alternative path, TC-03b
LOW_CONFIDENCE_MSG = "Low confidence prediction. Please consult a healthcare professional."
UNAVAILABLE_MSG = "Recommendation service is temporarily unavailable."
DISCLAIMER = ("This is an academic project, not a substitute for professional medical advice. "
              "Always consult a doctor or pharmacist before taking any medicine.")


# ---------------------------------------------------------------------------
# NLP symptom extraction (unchanged logic from the accepted prototype)
# ---------------------------------------------------------------------------
SYNONYMS = {
    "throwing up": "vomiting", "puking": "vomiting", "nauseous": "nausea",
    "feel sick": "nausea", "tummy ache": "stomach_pain", "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain", "high temperature": "high_fever", "temperature": "mild_fever",
    "fever": "mild_fever",
    "cant sleep": "insomnia", "can not sleep": "insomnia", "trouble sleeping": "insomnia",
    "runny nose": "runny_nose", "stuffy nose": "congestion", "blocked nose": "congestion",
    "sore throat": "throat_irritation", "swollen glands": "swelled_lymph_nodes",
    "swollen lymph nodes": "swelled_lymph_nodes", "cant smell": "loss_of_smell",
    "can not smell": "loss_of_smell", "tired": "fatigue",
    "exhausted": "fatigue", "dizzy": "dizziness", "out of breath": "breathlessness",
    "short of breath": "breathlessness", "yellow skin": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes", "itchy skin": "itching", "itchy": "itching",
    "rash": "skin_rash",
    # additions for the final system (everyday wording -> clinical column names)
    "breathless": "breathlessness", "difficulty breathing": "breathlessness", "cant breathe": "breathlessness",
    "chest hurts": "chest_pain", "chest tightness": "chest_pain", "pain in my chest": "chest_pain",
    "head hurts": "headache", "head ache": "headache", "migraine": "headache",
    "coughing": "cough", "loose motion": "diarrhoea", "loose motions": "diarrhoea", "diarrhea": "diarrhoea",
    "constipated": "constipation", "vomit": "vomiting", "vomited": "vomiting", "sneezing": "continuous_sneezing",
    "sweaty": "sweating", "body ache": "muscle_pain", "body aches": "muscle_pain", "body pain": "muscle_pain",
    "backache": "back_pain", "back ache": "back_pain", "neck ache": "neck_pain", "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite", "shivering": "chills", "weak": "fatigue", "weakness": "fatigue",
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
    """Free-text symptom extraction: curated synonyms, exact multi-word phrases,
    length-matched fuzzy fallback, then single-word matches.  Returns
    {symptom_key: matched surface text}."""
    phrase_bank = {c: _clean_phrase(c) for c in symptom_columns}
    text_norm = re.sub(r"\s+", " ", text.lower().strip())
    matched = {}
    words = _tokenize(text)
    used_idx = set()

    for phrase, target in SYNONYMS.items():
        if phrase in text_norm and target in phrase_bank and target not in matched:
            matched[target] = phrase

    remaining = [(k, p) for k, p in phrase_bank.items() if k not in matched]
    remaining.sort(key=lambda kp: len(kp[1].split()), reverse=True)

    for key, phrase in remaining:  # exact multi-word containment
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

    remaining = [(k, p) for k, p in remaining if k not in matched]
    for key, phrase in remaining:  # word-level fuzzy fallback
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

    # "high fever" already implies fever: do not also keep mild_fever that came from a bare
    # "fever"/"temperature" in the same message.  (Done after all stages, because the phrase
    # "high fever" is only matched in stage 2.)
    if "high_fever" in matched and matched.get("mild_fever") in ("fever", "temperature") \
            and "mild fever" not in text_norm:
        del matched["mild_fever"]
    return matched


# ---------------------------------------------------------------------------
# Optional profile read from the sentence (age group, gender, allergy, pregnancy)
# ---------------------------------------------------------------------------
_DURATION_BEFORE = re.compile(r"(?:for|since|last|past|over|about|around)\s*$")
_AGE_PATTERNS = [
    re.compile(r"\b(\d{1,3})\s*[- ]?\s*(?:years?|yrs?)\s*[- ]?\s*old\b"),
    re.compile(r"\b(\d{1,3})\s*(?:y/?o|yo)\b"),
    re.compile(r"\b(?:i am|i'm|im|age|aged|my age is)\s*(?:is\s*)?(\d{1,3})\b(?!\s*(?:days?|hours?|weeks?|months?|years?|times|%|degrees?))"),
]
_ALLERGEN_STOP = {"and", "an", "a", "the", "my", "skin", "food", "seasonal", "no", "with", "of", "i", "have", "to",
                  "any", "some", "also", "but", "or", "so", "it", "is", "am", "has", "had", "dust", "pollen"}
_ALLERGY_TO = re.compile(r"\b(?<!not )(?<!n't )(?:allergic|allergy|allergies)\s*(?:to|:|with)\s*([a-z0-9 ,/&\-]+)")
_ALLERGY_X = re.compile(r"\b([a-z]{4,})\s+allergy\b")
# words that end the allergen phrase and start the symptom description
_ALLERGY_END = re.compile(
    r"\s(?:with|and i|but i|but|who|having|having a|suffering|i have|i am|i'm|i feel|also|and have|and has|and my|"
    r"and a|and some|since|for)\b")


def age_group_from_age(age: int) -> str:
    if age < 13:
        return "child"
    if age < 18:
        return "teen"
    if age < 60:
        return "adult"
    return "senior"


def extract_profile(text: str, previous: Optional[dict] = None) -> Tuple[dict, str]:
    """Return (profile, text_without_profile_phrases).

    profile keys: age, age_group, gender, allergies (list), pregnant (bool).
    Values found in this message override those in `previous`.
    """
    profile = {"age": None, "age_group": None, "gender": None, "allergies": [], "pregnant": False}
    if previous:
        profile.update({k: (list(v) if isinstance(v, list) else v) for k, v in previous.items() if k in profile})
    low = " " + text.lower().replace("’", "'") + " "
    remaining = low

    # age
    for pat in _AGE_PATTERNS:
        for m in pat.finditer(low):
            if _DURATION_BEFORE.search(low[: m.start()].rstrip() + " ") and "old" not in m.group(0):
                continue
            age = int(m.group(1))
            if 0 < age < 120:
                profile["age"], profile["age_group"] = age, age_group_from_age(age)
                remaining = remaining.replace(m.group(0), " ")
                break
        if profile["age"] is not None and previous is None:
            break
    words = set(re.findall(r"[a-z]+", low))
    if profile["age"] is None:
        if words & {"child", "kid", "baby", "infant", "toddler", "son", "daughter"}:
            profile["age_group"] = "child"
        elif words & {"elderly", "senior"}:
            profile["age_group"] = "senior"

    # gender
    if words & {"female", "woman", "girl", "lady"}:
        profile["gender"] = "female"
    elif words & {"male", "man", "boy"}:
        profile["gender"] = "male"
    if words & {"pregnant", "pregnancy", "expecting"}:
        profile["pregnant"] = True
        profile["gender"] = profile["gender"] or "female"
    for w in ("female", "male", "woman", "man", "girl", "boy", "lady", "pregnant"):
        remaining = re.sub(rf"\b{w}\b", " ", remaining)

    # allergies
    found = []
    m = _ALLERGY_TO.search(low)
    if m:
        chunk = m.group(1)
        end = _ALLERGY_END.search(" " + chunk)
        if end:
            chunk = (" " + chunk)[: end.start()]
        consumed = [m.group(0)[: m.start(1) - m.start(0)]]  # the words "allergic to"
        for i, part in enumerate(re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", chunk)):
            toks = [t for t in part.split() if t not in _ALLERGEN_STOP]
            if not toks:
                continue
            cand = " ".join(toks[:2])
            # the first allergen is always taken; later list items only if they are known allergens
            # (otherwise "allergic to penicillin, I have fever, cough" would swallow the symptoms)
            if i == 0 or cand in ref.ALLERGY_MAP or toks[0] in ref.ALLERGY_MAP:
                found.append(cand if cand in ref.ALLERGY_MAP or len(toks) == 1 else toks[0] if toks[0] in ref.ALLERGY_MAP else cand)
                consumed.append(part)
            else:
                break
        for c in consumed:
            if c.strip():
                remaining = remaining.replace(c, " ", 1)
    for a in found:
        if a and a not in profile["allergies"]:
            profile["allergies"].append(a)
    return profile, re.sub(r"\s+", " ", remaining).strip()


# ---------------------------------------------------------------------------
# Active model (loaded from the registry, FR-12, UC-07)
# ---------------------------------------------------------------------------
class ModelUnavailable(RuntimeError):
    pass


_model_cache: Dict[tuple, dict] = {}


def _load_any(path: str) -> dict:
    p = Path(path)
    if p.name == "rf_model.pkl":  # prototype artefacts: three separate files
        import json, pickle
        with open(p, "rb") as f:
            model = pickle.load(f)
        with open(p.parent / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        cols = json.load(open(p.parent / "symptom_columns.json"))
        return {"model": model, "label_encoder": le, "symptom_columns": cols, "metadata": {"name": "prototype"}}
    return trainer.load_bundle(path)


def _quick_rebuild(reason: str) -> Tuple[dict, str]:
    """Fallback when a stored model cannot be unpickled (e.g. scikit-learn version
    mismatch): retrain a Random Forest from the training CSV in a few seconds."""
    import sklearn
    db.log_event("model_rebuild", reason, level="WARNING")
    X, y, _ = trainer.clean_dataset(trainer.load_dataset("Datasets/Training.csv"))
    run = trainer.train_and_compare(X, y, models=["Random Forest"], tune=False)
    db.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = str(db.MODEL_DIR / f"auto_random_forest_sk{sklearn.__version__}.pkl")
    trainer.save_bundle(path, run["results"]["Random Forest"]["estimator"], run["label_encoder"],
                        run["symptom_columns"], {"name": "Random Forest (auto-rebuilt)",
                                                 "metrics": trainer.json_safe_metrics(run, "Random Forest")})
    return trainer.load_bundle(path), path


def load_active_model() -> Tuple[dict, dict]:
    """Return (bundle, registry_row) for the production model."""
    row = db.production_model()
    if row is None:
        raise ModelUnavailable("No production model registered.")
    path = row["file_path"]
    try:
        key = (path, Path(path).stat().st_mtime)
    except OSError as exc:
        raise ModelUnavailable(f"Model file missing: {path}") from exc
    if key in _model_cache:
        return _model_cache[key], row
    try:
        bundle = _load_any(path)
    except Exception as exc:
        bundle, new_path = _quick_rebuild(f"{path}: {exc}")
        db.execute("UPDATE ml_models SET file_path=? WHERE model_id=?", (new_path, row["model_id"]))
        key = (new_path, Path(new_path).stat().st_mtime)
        row = db.production_model()
    _model_cache.clear()
    _model_cache[key] = bundle
    return bundle, row


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    status: str = "ok"                       # ok | no_symptoms | error
    message: str = ""
    symptoms: Dict[str, str] = field(default_factory=dict)
    profile: dict = field(default_factory=dict)
    disease: str = ""
    disease_id: Optional[int] = None
    confidence: float = 0.0
    top3: List[Tuple[str, float]] = field(default_factory=list)
    description: str = ""
    medicines: List[dict] = field(default_factory=list)
    precautions: List[str] = field(default_factory=list)
    diet: List[str] = field(default_factory=list)
    workout: List[str] = field(default_factory=list)
    low_confidence: bool = False
    notes: List[str] = field(default_factory=list)
    model_version: str = ""
    model_id: Optional[int] = None
    query_id: Optional[int] = None
    fever_ambiguous: bool = False


def rank_medicines(top3_ids: List[Tuple[int, float]], profile: dict) -> List[dict]:
    """FR-24: rank the predicted disease's treatments by relevance.

    relevance(m) = sum over the top-3 candidate diseases d of  P(d) * prior(d, m)
    where prior is the list-position weight stored in disease_medicine_map.
    A treatment that also appears for the 2nd/3rd most likely disease therefore
    ranks higher than one that is specific to the top disease only.
    """
    main_id = top3_ids[0][0]
    scores: Dict[int, float] = {}
    likely_for: Dict[int, List[str]] = {}  # which of the likely diseases list this treatment
    for did, p in top3_ids:
        dname = db.q1("SELECT name FROM diseases WHERE disease_id=?", (did,))["name"]
        for m in db.disease_medicines(did):
            scores[m["medicine_id"]] = scores.get(m["medicine_id"], 0.0) + p * m["relevance_score"]
            likely_for.setdefault(m["medicine_id"], []).append(dname)
    out = []
    for m in db.disease_medicines(main_id):
        name = m["name"]
        conflicts = ref.allergy_conflicts(profile.get("allergies", []), name)
        out.append({
            "medicine_id": m["medicine_id"], "name": name, "flag": m["prescription_flag"],
            "dosage_note": m["dosage_note"], "used_for": likely_for.get(m["medicine_id"], [])[:3],
            "relevance": round(min(scores[m["medicine_id"]], 1.0) * 100, 1), "allergy": conflicts,
            "position": m["list_position"],
        })
    out.sort(key=lambda x: (-x["relevance"], x["position"]))
    return out


def _predict(bundle: dict, detected: Dict[str, str], rec: "Recommendation") -> np.ndarray:
    """Probability vector for the detected symptoms.

    A bare "fever" does not say whether it is mild or high.  Only 3 diseases in
    the data have `mild_fever`, but many have `high_fever`, so mapping every
    "fever" to `mild_fever` would bias the result towards those 3 diseases.
    Instead the two readings are averaged, weighted by how many diseases list
    each of them.
    """
    cols = bundle["symptom_columns"]
    keys = list(detected)
    bare_fever = detected.get("mild_fever") in ("fever", "temperature") and "high_fever" not in detected
    if bare_fever and "high_fever" in cols and "mild_fever" in cols:
        n_mild = max(db.symptom_disease_count("mild_fever"), 1)
        n_high = max(db.symptom_disease_count("high_fever"), 1)
        w_high = n_high / (n_mild + n_high)
        p_mild = trainer.predict_proba_from_symptoms(bundle, keys)
        p_high = trainer.predict_proba_from_symptoms(bundle, [k if k != "mild_fever" else "high_fever" for k in keys])
        rec.fever_ambiguous = True
        return (1 - w_high) * p_mild + w_high * p_high
    return trainer.predict_proba_from_symptoms(bundle, keys)


def recommend(text: str, previous_profile: Optional[dict] = None) -> Recommendation:
    """UC-03 steps 5-8 (history logging is done by the caller)."""
    rec = Recommendation()
    try:
        bundle, row = load_active_model()
    except Exception as exc:
        db.log_event("model_unavailable", str(exc), level="ERROR")
        rec.status, rec.message = "error", UNAVAILABLE_MSG
        return rec

    profile, remaining_text = extract_profile(text, previous_profile)
    rec.profile = profile
    cols = bundle["symptom_columns"]
    detected = extract_symptoms(remaining_text, cols)
    rec.symptoms = detected
    if not detected:
        if profile != (previous_profile or {}) and any([profile["age_group"], profile["gender"], profile["allergies"], profile["pregnant"]]):
            rec.status = "profile_only"
            rec.message = (f"Thanks - I noted: {describe_profile(profile)}. "
                           "Now tell me which symptoms you have, in your own words.")
        else:
            rec.status = "no_symptoms"
            rec.message = ("I couldn't confidently pick out any symptoms from that - could you try describing them "
                           "more specifically? For example: \"I have itching, joint pain, and a high temperature.\"")
        return rec

    try:
        proba = _predict(bundle, detected, rec)
        le = bundle["label_encoder"]
        order = np.argsort(proba)[::-1][:3]
        top3 = [(db.display_name(le.classes_[i]), float(proba[i]) * 100) for i in order]
        disease = top3[0][0]
        drow = db.disease_by_name(disease)
        if drow is None:
            raise LookupError(f"Disease '{disease}' is not in the reference database.")
        top_ids = []
        for name, conf in top3:
            r = db.disease_by_name(name)
            if r is not None:
                top_ids.append((r["disease_id"], conf / 100))
        meds = rank_medicines(top_ids, profile)
    except Exception as exc:
        db.log_event("inference_failed", repr(exc), level="ERROR")
        rec.status, rec.message = "error", UNAVAILABLE_MSG
        return rec

    guide = db.guidance(drow["disease_id"])
    rec.disease, rec.disease_id, rec.confidence, rec.top3 = disease, drow["disease_id"], top3[0][1], top3
    rec.description = drow["description"] or ""
    rec.medicines = meds
    rec.precautions, rec.diet, rec.workout = guide["Precaution"], guide["Diet"], guide["Workout"]
    rec.model_version, rec.model_id = row["version"], row["model_id"]
    rec.low_confidence = rec.confidence < LOW_CONFIDENCE_THRESHOLD

    if rec.low_confidence:
        rec.notes.append(LOW_CONFIDENCE_MSG)
        rec.notes.append(
            f"Only {len(detected)} symptom(s) were detected, so the match is uncertain. Describe more detail "
            "(how long, what else you feel) for a more reliable result.")
    elif len(detected) == 1:
        rec.notes.append("I only picked up one symptom - feel free to describe more for a more confident prediction.")
    if rec.fever_ambiguous:
        rec.notes.append("You mentioned fever without saying how high, so I considered both mild and high fever. "
                         "Tell me the temperature or say \"high fever\" / \"mild fever\" for a sharper result.")
    if profile.get("age_group") in ("child", "teen"):
        rec.notes.append("Medicine doses for children and teenagers differ from adult doses. "
                         "Ask a doctor or pharmacist before giving any medicine.")
    if profile.get("age_group") == "senior":
        rec.notes.append("Older adults are more sensitive to many medicines. Check dose and interactions with a pharmacist.")
    if profile.get("pregnant"):
        rec.notes.append("You mentioned pregnancy: do not take any medicine without asking your doctor first.")
    flagged = [m for m in meds if m["allergy"]]
    if flagged:
        rec.notes.append("Allergy warning: " + ", ".join(
            f"{m['name']} (you mentioned {', '.join(m['allergy'])})" for m in flagged)
            + " may conflict with your stated allergy. Do not take it without medical advice.")
    return rec


def describe_profile(profile: dict) -> str:
    prof = []
    if profile.get("age_group"):
        prof.append(f"age group: {profile['age_group']}" + (f" ({profile['age']})" if profile.get("age") else ""))
    if profile.get("gender"):
        prof.append(f"gender: {profile['gender']}")
    if profile.get("allergies"):
        prof.append("allergies: " + ", ".join(profile["allergies"]))
    if profile.get("pregnant"):
        prof.append("pregnant")
    return "; ".join(prof)


def to_title(sym_key: str) -> str:
    return sym_key.replace("_", " ").title()


def format_reply(rec: Recommendation) -> str:
    """Markdown shown in the chat bubble."""
    if rec.status != "ok":
        return rec.message
    L = []
    names = [("Fever (severity not stated)" if (k == "mild_fever" and rec.fever_ambiguous) else to_title(k))
             for k in rec.symptoms]
    L.append(f"**Detected symptoms:** {', '.join(names)}")
    prof = describe_profile(rec.profile)
    if prof:
        L.append("**Noted from your message:** " + prof)
    L.append(f"\n### Predicted Condition: {rec.disease}")
    L.append(f"**Confidence:** {rec.confidence:.1f}%")
    if rec.low_confidence:
        L.append(f"\n> ⚠️ **{LOW_CONFIDENCE_MSG}**")
    L.append(f"\n{rec.description}")

    L.append("\n**Suggested medicines and treatments (ranked by relevance)**")
    L.append("\n| # | Medicine / treatment | Relevance | Prescription | Commonly listed for | Dosage note |")
    L.append("|---|---|---|---|---|---|")
    for i, m in enumerate(rec.medicines, 1):
        warn = " ⚠️" if m["allergy"] else ""
        L.append(f"| {i} | {m['name']}{warn} | {m['relevance']:.0f}% | {ref.FLAG_ICONS[m['flag']]} | "
                 f"{', '.join(m['used_for'])} | {m['dosage_note']} |")
    L.append("\n**Precautions**")
    L += [f"- {p}" for p in rec.precautions]
    L.append("\n**Suggested Diet**")
    L += [f"- {d}" for d in rec.diet]
    L.append("\n**Suggested Workout**")
    L += [f"- {w}" for w in rec.workout]
    if rec.notes:
        L.append("")
        L += [f"*{n}*" for n in rec.notes if n != LOW_CONFIDENCE_MSG]
    L.append(f"\n---\n*{DISCLAIMER}*")
    return "\n".join(L)
