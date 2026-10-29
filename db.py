"""
db.py - SQLite data tier (Design Document section 6, Database Design).

Tables from the Design Document: users, diseases, symptoms, disease_symptom_map,
medicines, disease_medicine_map, care_guidance, recommendation_queries,
recommendation_details, feedback, datasets, ml_models.
Supporting tables added during development: sessions, password_resets, audit_log.

Sensitive columns (symptom text, profile, feedback comment) are AES-256-GCM
encrypted by security.py before they are stored (NFR-04).
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import reference_data as ref
import security

DB_PATH = Path(os.environ.get("MRS_DB_PATH", "data/mrs.db"))
DATASET_DIR = Path(os.environ.get("MRS_DATASET_DIR", "data/datasets"))
DATA_FILES = Path("Datasets")
MODEL_DIR = Path(os.environ.get("MRS_MODEL_DIR", "Models/versions"))

ROLES = ("patient", "provider", "pharmacist", "admin")
MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 15
SESSION_HOURS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    password_salt   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('patient','provider','pharmacist','admin')),
    is_verified     INTEGER NOT NULL DEFAULT 0,
    is_active       INTEGER NOT NULL DEFAULT 1,
    consent_given   INTEGER NOT NULL DEFAULT 0,
    consent_at      TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    created_at      TEXT NOT NULL,
    last_login      TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS password_resets (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS diseases (
    disease_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE IF NOT EXISTS symptoms (
    symptom_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    severity   INTEGER
);
CREATE TABLE IF NOT EXISTS disease_symptom_map (
    disease_id INTEGER NOT NULL REFERENCES diseases(disease_id) ON DELETE CASCADE,
    symptom_id INTEGER NOT NULL REFERENCES symptoms(symptom_id) ON DELETE CASCADE,
    PRIMARY KEY (disease_id, symptom_id)
);
CREATE TABLE IF NOT EXISTS medicines (
    medicine_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT NOT NULL UNIQUE,
    prescription_flag     TEXT NOT NULL CHECK (prescription_flag IN ('OTC','Rx/OTC','Rx','Self-care','Clinical')),
    dosage_note           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS disease_medicine_map (
    disease_id      INTEGER NOT NULL REFERENCES diseases(disease_id) ON DELETE CASCADE,
    medicine_id     INTEGER NOT NULL REFERENCES medicines(medicine_id) ON DELETE CASCADE,
    relevance_score REAL NOT NULL,
    list_position   INTEGER NOT NULL,
    PRIMARY KEY (disease_id, medicine_id)
);
CREATE TABLE IF NOT EXISTS care_guidance (
    guidance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_id  INTEGER NOT NULL REFERENCES diseases(disease_id) ON DELETE CASCADE,
    type        TEXT NOT NULL CHECK (type IN ('Precaution','Diet','Workout')),
    text        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    filename           TEXT NOT NULL,
    stored_path        TEXT NOT NULL,
    uploaded_by        INTEGER REFERENCES users(user_id),
    uploaded_at        TEXT NOT NULL,
    n_rows             INTEGER,
    n_columns          INTEGER,
    n_symptoms         INTEGER,
    n_diseases         INTEGER,
    duplicates_removed INTEGER,
    status             TEXT NOT NULL DEFAULT 'validated'
);
CREATE TABLE IF NOT EXISTS ml_models (
    model_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT NOT NULL,
    algorithm   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    dataset_id  INTEGER REFERENCES datasets(dataset_id),
    accuracy    REAL, precision_w REAL, recall_w REAL, f1_w REAL,
    acc_3_symptoms REAL, cv_accuracy REAL,
    params_json TEXT,
    status      TEXT NOT NULL CHECK (status IN ('candidate','production','archived','rejected')),
    trained_at  TEXT,
    deployed_at TEXT,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS recommendation_queries (
    query_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at           TEXT NOT NULL,
    input_text_enc       TEXT NOT NULL,
    symptoms_enc         TEXT NOT NULL,
    age_group_enc        TEXT,
    gender_enc           TEXT,
    allergy_enc          TEXT,
    predicted_disease_id INTEGER REFERENCES diseases(disease_id),
    confidence           REAL,
    low_confidence       INTEGER NOT NULL DEFAULT 0,
    model_id             INTEGER REFERENCES ml_models(model_id)
);
CREATE TABLE IF NOT EXISTS recommendation_details (
    detail_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        INTEGER NOT NULL REFERENCES recommendation_queries(query_id) ON DELETE CASCADE,
    medicine_id     INTEGER NOT NULL REFERENCES medicines(medicine_id),
    rank_order      INTEGER NOT NULL,
    relevance_score REAL NOT NULL,
    allergy_flag    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    INTEGER NOT NULL UNIQUE REFERENCES recommendation_queries(query_id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment_enc TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    log_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    level   TEXT NOT NULL,
    event   TEXT NOT NULL,
    user_id INTEGER,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_user ON recommendation_queries(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_details_query ON recommendation_details(query_id);
CREATE INDEX IF NOT EXISTS idx_guidance_disease ON care_guidance(disease_id);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# connection helpers
# ---------------------------------------------------------------------------
@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def q(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    with connect() as c:
        return c.execute(sql, params).fetchall()


def q1(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    with connect() as c:
        return c.execute(sql, params).fetchone()


def execute(sql: str, params: tuple = ()) -> int:
    with connect() as c:
        return c.execute(sql, params).lastrowid


def log_event(event: str, detail: str = "", level: str = "INFO", user_id: Optional[int] = None) -> None:
    """Audit / error log (UC exceptions: 'logs the incident for admin review')."""
    try:
        execute("INSERT INTO audit_log(ts, level, event, user_id, detail) VALUES (?,?,?,?,?)",
                (now(), level, event, user_id, detail[:2000]))
    except Exception:
        pass  # logging must never break the app


# ---------------------------------------------------------------------------
# initialisation + seeding of reference data from the CSV files
# ---------------------------------------------------------------------------
def normalize(name: str) -> str:
    return " ".join(str(name).split()).strip().lower()


DISEASE_ALIASES = {"peptic ulcer diseae": "peptic ulcer disease"}  # spelling error in the source data
DISPLAY_NAMES = {"peptic ulcer disease": "Peptic ulcer disease"}


def display_name(name: str) -> str:
    """Corrected, whitespace-normalised disease name (fixes 'diseae' typo in the source data)."""
    k = DISEASE_ALIASES.get(normalize(name), normalize(name))
    return DISPLAY_NAMES.get(k, " ".join(str(name).split()))


def init_db(force: bool = False) -> None:
    """Create tables and seed reference data, the admin account and model versions."""
    with connect() as c:
        c.executescript(SCHEMA)
    if not force and q1("SELECT 1 FROM diseases LIMIT 1"):
        return
    seed_reference_data()
    seed_admin()
    seed_models()
    log_event("db_initialised", "reference data, admin account and model registry seeded")


def seed_reference_data(data_dir: Path = DATA_FILES) -> dict:
    training = pd.read_csv(data_dir / "Training.csv")
    description = pd.read_csv(data_dir / "description.csv")
    precautions = pd.read_csv(data_dir / "precautions_df.csv")
    medications = pd.read_csv(data_dir / "medications.csv")
    diets = pd.read_csv(data_dir / "diets.csv")
    workout = pd.read_csv(data_dir / "workout_df.csv")
    severity = pd.read_csv(data_dir / "Symptom-severity.csv")

    sym_cols = [c for c in training.columns if c != "prognosis" and not c.startswith("Unnamed")]
    sev = {normalize(r.iloc[0]).replace(" ", "_"): int(r.iloc[1]) for _, r in severity.iterrows()}

    def key(d):
        k = normalize(d)
        return DISEASE_ALIASES.get(k, k)

    desc = {key(r["Disease"]): r["Description"] for _, r in description.iterrows()}
    canonical = {key(d): display_name(d) for d in training["prognosis"].unique()}
    counts = {"diseases": 0, "symptoms": 0, "medicines": 0, "guidance": 0}

    with connect() as c:
        for sname in sym_cols:
            c.execute("INSERT OR IGNORE INTO symptoms(name, severity) VALUES (?,?)",
                      (sname.strip(), sev.get(normalize(sname).replace(" ", "_"))))
        counts["symptoms"] = len(sym_cols)
        disease_id = {}
        for k, name in canonical.items():
            cur = c.execute("INSERT OR IGNORE INTO diseases(name, description) VALUES (?,?)",
                            (name, desc.get(k, "No description available.")))
            disease_id[k] = c.execute("SELECT disease_id FROM diseases WHERE name=?", (name,)).fetchone()[0]
        counts["diseases"] = len(disease_id)

        # disease <-> symptom map from the training data
        sym_id = {r["name"]: r["symptom_id"] for r in c.execute("SELECT * FROM symptoms")}
        for dname, grp in training.drop_duplicates().groupby("prognosis"):
            did = disease_id[key(dname)]
            present = grp[sym_cols].max()
            for sname in present[present > 0].index:
                c.execute("INSERT OR IGNORE INTO disease_symptom_map VALUES (?,?)", (did, sym_id[sname.strip()]))

        # medicines + disease <-> medicine map (relevance prior from list position)
        for _, row in medications.iterrows():
            did = disease_id.get(key(row["Disease"]))
            if did is None:
                continue
            try:
                meds = ast.literal_eval(row["Medication"])
            except (ValueError, SyntaxError):
                meds = [row["Medication"]]
            for pos, m in enumerate(meds):
                flag = ref.prescription_flag(m)
                c.execute("INSERT OR IGNORE INTO medicines(name, prescription_flag, dosage_note) VALUES (?,?,?)",
                          (m.strip(), flag, ref.DOSAGE_NOTES[flag]))
                mid = c.execute("SELECT medicine_id FROM medicines WHERE name=?", (m.strip(),)).fetchone()[0]
                c.execute("INSERT OR REPLACE INTO disease_medicine_map VALUES (?,?,?,?)",
                          (did, mid, round(max(1.0 - 0.05 * pos, 0.5), 2), pos + 1))
        counts["medicines"] = c.execute("SELECT COUNT(*) FROM medicines").fetchone()[0]

        # care guidance: precautions, diet, workout
        for _, row in precautions.iterrows():
            did = disease_id.get(key(row["Disease"]))
            for col in [x for x in precautions.columns if "Precaution" in x]:
                if did and pd.notna(row[col]) and str(row[col]).strip():
                    c.execute("INSERT INTO care_guidance(disease_id,type,text) VALUES (?,?,?)",
                              (did, "Precaution", str(row[col]).strip()))
        for _, row in diets.iterrows():
            did = disease_id.get(key(row["Disease"]))
            try:
                items = ast.literal_eval(row["Diet"])
            except (ValueError, SyntaxError):
                items = [row["Diet"]]
            for it in items:
                if did:
                    c.execute("INSERT INTO care_guidance(disease_id,type,text) VALUES (?,?,?)", (did, "Diet", str(it)))
        for _, row in workout.iterrows():
            did = disease_id.get(key(row["disease"]))
            if did and pd.notna(row["workout"]):
                c.execute("INSERT INTO care_guidance(disease_id,type,text) VALUES (?,?,?)",
                          (did, "Workout", str(row["workout"])))
        counts["guidance"] = c.execute("SELECT COUNT(*) FROM care_guidance").fetchone()[0]
    return counts


DEFAULT_ADMIN_EMAIL = os.environ.get("MRS_ADMIN_EMAIL", "admin@mrs.local")
DEFAULT_ADMIN_PASSWORD = os.environ.get("MRS_ADMIN_PASSWORD", "Admin@12345")


def seed_admin() -> None:
    if q1("SELECT 1 FROM users WHERE email=?", (DEFAULT_ADMIN_EMAIL,)):
        return
    h, s = security.hash_password(DEFAULT_ADMIN_PASSWORD)
    execute("""INSERT INTO users(full_name,email,password_hash,password_salt,role,is_verified,consent_given,consent_at,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("System Administrator", DEFAULT_ADMIN_EMAIL, h, s, "admin", 1, 1, now(), now()))


def seed_models() -> None:
    """Register the prototype model (v1, archived baseline) and the notebook's selected model (v2, production)."""
    if q1("SELECT 1 FROM ml_models LIMIT 1"):
        return
    summary_path = Path("Models/model_comparison.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    base = summary.get("baseline_prototype_rf", {})
    bundles = sorted(Path("Models/versions").glob("model_v2_*.pkl"))
    # v1: the accepted prototype model (kept for rollback / comparison)
    if Path("Models/rf_model.pkl").exists():
        execute("""INSERT INTO ml_models(version,algorithm,file_path,accuracy,precision_w,recall_w,f1_w,acc_3_symptoms,
                   cv_accuracy,params_json,status,trained_at,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("v1", "Random Forest (prototype)", "Models/rf_model.pkl", base.get("accuracy"), base.get("precision"),
                 base.get("recall"), base.get("f1"), base.get("acc_3_symptoms"), None,
                 json.dumps({"n_estimators": 200}), "archived", None,
                 "Prototype model trained on raw data without augmentation. Metrics re-evaluated on the de-duplicated test split."))
    if bundles:
        name = summary.get("selected", "Random Forest")
        m = summary.get("models", {}).get(name, {})
        execute("""INSERT INTO ml_models(version,algorithm,file_path,accuracy,precision_w,recall_w,f1_w,acc_3_symptoms,
                   cv_accuracy,params_json,status,trained_at,deployed_at,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("v2", name, str(bundles[0]).replace("\\", "/"), m.get("test_accuracy"), m.get("test_precision"),
                 m.get("test_recall"), m.get("test_f1"), m.get("test_acc_3_symptoms"), m.get("cv_accuracy_mean"),
                 json.dumps(m.get("best_params", {})), "production", now(), now(),
                 "Selected by model_comparison.ipynb (de-duplicated data, symptom-subset augmentation, GridSearchCV)."))
    elif Path("Models/rf_model.pkl").exists():
        execute("UPDATE ml_models SET status='production', deployed_at=? WHERE version='v1'", (now(),))


# ---------------------------------------------------------------------------
# users, sessions, resets
# ---------------------------------------------------------------------------
def create_user(full_name, email, password, role, consent=True, verified=True) -> int:
    h, s = security.hash_password(password)
    return execute(
        """INSERT INTO users(full_name,email,password_hash,password_salt,role,is_verified,consent_given,consent_at,created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (full_name.strip(), email.strip().lower(), h, s, role, int(verified), int(consent), now() if consent else None, now()))


def get_user_by_email(email: str):
    return q1("SELECT * FROM users WHERE email=?", ((email or "").strip().lower(),))


def get_user(user_id: int):
    return q1("SELECT * FROM users WHERE user_id=?", (user_id,))


def list_users():
    return q("SELECT user_id, full_name, email, role, is_active, is_verified, failed_attempts, locked_until, created_at, last_login FROM users ORDER BY user_id")


def create_session(user_id: int) -> str:
    token = security.new_token()
    expires = (datetime.now() + timedelta(hours=SESSION_HOURS)).isoformat(timespec="seconds")
    execute("INSERT INTO sessions VALUES (?,?,?,?)", (security.token_hash(token), user_id, now(), expires))
    return token


def session_user(token: Optional[str]):
    if not token:
        return None
    row = q1("""SELECT u.* FROM sessions s JOIN users u ON u.user_id=s.user_id
                WHERE s.token_hash=? AND s.expires_at > ? AND u.is_active=1""", (security.token_hash(token), now()))
    return row


def end_session(token: Optional[str]) -> None:
    if token:
        execute("DELETE FROM sessions WHERE token_hash=?", (security.token_hash(token),))


# ---------------------------------------------------------------------------
# reference data access (used by the recommendation engine)
# ---------------------------------------------------------------------------
def disease_by_name(name: str):
    return q1("SELECT * FROM diseases WHERE lower(name)=lower(?)", (display_name(name),))


def disease_medicines(disease_id: int):
    return q("""SELECT m.medicine_id, m.name, m.prescription_flag, m.dosage_note, dm.relevance_score, dm.list_position
                FROM disease_medicine_map dm JOIN medicines m ON m.medicine_id=dm.medicine_id
                WHERE dm.disease_id=? ORDER BY dm.list_position""", (disease_id,))


def symptom_disease_count(symptom_name: str) -> int:
    """How many diseases in the reference data list this symptom (used to weight ambiguous input)."""
    r = q1("""SELECT COUNT(*) FROM disease_symptom_map dm JOIN symptoms s ON s.symptom_id=dm.symptom_id
              WHERE s.name=?""", (symptom_name,))
    return int(r[0])


def medicine_uses(medicine_id: int, limit: int = 3) -> List[str]:
    rows = q("""SELECT d.name FROM disease_medicine_map dm JOIN diseases d ON d.disease_id=dm.disease_id
                WHERE dm.medicine_id=? ORDER BY dm.list_position LIMIT ?""", (medicine_id, limit))
    return [r["name"] for r in rows]


def guidance(disease_id: int) -> Dict[str, List[str]]:
    out = {"Precaution": [], "Diet": [], "Workout": []}
    for r in q("SELECT type, text FROM care_guidance WHERE disease_id=? ORDER BY guidance_id", (disease_id,)):
        out[r["type"]].append(r["text"])
    return out


# ---------------------------------------------------------------------------
# queries (history) and feedback  - sensitive fields are encrypted
# ---------------------------------------------------------------------------
def save_query(user_id, text, symptoms: List[str], profile: dict, disease_id, confidence, low_conf, model_id,
               medicines: List[dict]) -> int:
    with connect() as c:
        cur = c.execute(
            """INSERT INTO recommendation_queries(user_id,created_at,input_text_enc,symptoms_enc,age_group_enc,gender_enc,
               allergy_enc,predicted_disease_id,confidence,low_confidence,model_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (user_id, now(), security.encrypt_text(text), security.encrypt_text(json.dumps(symptoms)),
             security.encrypt_text(profile.get("age_group")), security.encrypt_text(profile.get("gender")),
             security.encrypt_text(", ".join(profile.get("allergies", [])) or None),
             disease_id, confidence, int(low_conf), model_id))
        qid = cur.lastrowid
        for i, m in enumerate(medicines, 1):
            c.execute("""INSERT INTO recommendation_details(query_id,medicine_id,rank_order,relevance_score,allergy_flag)
                         VALUES (?,?,?,?,?)""", (qid, m["medicine_id"], i, m["relevance"], int(bool(m.get("allergy")))))
    return qid


def user_history(user_id: int) -> List[Dict[str, Any]]:
    rows = q("""SELECT rq.*, d.name AS disease, f.rating FROM recommendation_queries rq
                LEFT JOIN diseases d ON d.disease_id=rq.predicted_disease_id
                LEFT JOIN feedback f ON f.query_id=rq.query_id
                WHERE rq.user_id=? ORDER BY rq.created_at DESC, rq.query_id DESC""", (user_id,))
    out = []
    for r in rows:
        top = q1("""SELECT m.name FROM recommendation_details rd JOIN medicines m ON m.medicine_id=rd.medicine_id
                    WHERE rd.query_id=? ORDER BY rd.rank_order LIMIT 1""", (r["query_id"],))
        out.append({
            "query_id": r["query_id"], "created_at": r["created_at"], "disease": r["disease"],
            "confidence": r["confidence"], "low_confidence": bool(r["low_confidence"]), "rating": r["rating"],
            "input_text": security.decrypt_text(r["input_text_enc"]),
            "symptoms": json.loads(security.decrypt_text(r["symptoms_enc"]) or "[]"),
            "age_group": security.decrypt_text(r["age_group_enc"]), "gender": security.decrypt_text(r["gender_enc"]),
            "allergy": security.decrypt_text(r["allergy_enc"]),
            "top_medicine": top["name"] if top else None,
        })
    return out


def query_medicines(query_id: int):
    return q("""SELECT m.name, m.prescription_flag, m.dosage_note, rd.rank_order, rd.relevance_score, rd.allergy_flag
                FROM recommendation_details rd JOIN medicines m ON m.medicine_id=rd.medicine_id
                WHERE rd.query_id=? ORDER BY rd.rank_order""", (query_id,))


def save_feedback(query_id: int, user_id: int, rating: int, comment: str = "") -> None:
    if not 1 <= int(rating) <= 5:
        raise ValueError("Rating must be between 1 and 5.")
    execute("""INSERT INTO feedback(query_id,user_id,rating,comment_enc,created_at) VALUES (?,?,?,?,?)
               ON CONFLICT(query_id) DO UPDATE SET rating=excluded.rating, comment_enc=excluded.comment_enc, created_at=excluded.created_at""",
            (query_id, user_id, int(rating), security.encrypt_text(comment.strip() or None), now()))


def get_feedback(query_id: int):
    r = q1("SELECT * FROM feedback WHERE query_id=?", (query_id,))
    return r


# ---------------------------------------------------------------------------
# datasets and models (admin)
# ---------------------------------------------------------------------------
def add_dataset(filename, stored_path, uploaded_by, report: dict) -> int:
    return execute("""INSERT INTO datasets(filename,stored_path,uploaded_by,uploaded_at,n_rows,n_columns,n_symptoms,n_diseases,
                      duplicates_removed) VALUES (?,?,?,?,?,?,?,?,?)""",
                   (filename, str(stored_path), uploaded_by, now(), report.get("rows_in"), report.get("cols_in"),
                    report.get("n_symptoms"), report.get("n_diseases"), report.get("duplicates_removed")))


def list_datasets():
    return q("SELECT * FROM datasets ORDER BY dataset_id DESC")


def get_dataset(dataset_id: int):
    return q1("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,))


def list_models():
    return q("SELECT * FROM ml_models ORDER BY model_id DESC")


def production_model():
    return q1("SELECT * FROM ml_models WHERE status='production' ORDER BY model_id DESC LIMIT 1")


def next_version() -> str:
    r = q1("SELECT COUNT(*) AS n FROM ml_models")
    return f"v{r['n'] + 1}"


def add_model(version, algorithm, file_path, dataset_id, m: dict, params: dict, status="candidate", notes="") -> int:
    return execute("""INSERT INTO ml_models(version,algorithm,file_path,dataset_id,accuracy,precision_w,recall_w,f1_w,
                      acc_3_symptoms,cv_accuracy,params_json,status,trained_at,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (version, algorithm, str(file_path).replace("\\", "/"), dataset_id, m.get("test_accuracy"),
                    m.get("test_precision"), m.get("test_recall"), m.get("test_f1"), m.get("test_acc_3_symptoms"),
                    m.get("cv_accuracy_mean"), json.dumps(params, default=str), status, now(), notes))


def deploy_model(model_id: int) -> None:
    """Make model_id the production model and archive the previous one (UC-07 step 7)."""
    with connect() as c:
        c.execute("UPDATE ml_models SET status='archived' WHERE status='production'")
        c.execute("UPDATE ml_models SET status='production', deployed_at=? WHERE model_id=?", (now(), model_id))


def set_model_status(model_id: int, status: str) -> None:
    execute("UPDATE ml_models SET status=? WHERE model_id=?", (status, model_id))


# ---------------------------------------------------------------------------
# reports (admin)
# ---------------------------------------------------------------------------
def report_summary() -> dict:
    one = lambda sql: q1(sql)[0]
    return {
        "users": one("SELECT COUNT(*) FROM users"),
        "queries": one("SELECT COUNT(*) FROM recommendation_queries"),
        "low_confidence": one("SELECT COUNT(*) FROM recommendation_queries WHERE low_confidence=1"),
        "feedback_count": one("SELECT COUNT(*) FROM feedback"),
        "avg_rating": one("SELECT ROUND(AVG(rating),2) FROM feedback"),
        "avg_confidence": one("SELECT ROUND(AVG(confidence),1) FROM recommendation_queries"),
    }


def top_diseases(limit: int = 10):
    return q("""SELECT d.name AS disease, COUNT(*) AS queries, ROUND(AVG(rq.confidence),1) AS avg_confidence
                FROM recommendation_queries rq JOIN diseases d ON d.disease_id=rq.predicted_disease_id
                GROUP BY d.name ORDER BY queries DESC LIMIT ?""", (limit,))


def rating_distribution():
    return q("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating ORDER BY rating")


def recent_log(limit: int = 100):
    return q("SELECT * FROM audit_log ORDER BY log_id DESC LIMIT ?", (limit,))
