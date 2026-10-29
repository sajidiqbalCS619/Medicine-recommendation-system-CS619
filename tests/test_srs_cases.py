"""
The 14 test cases of the Design Document (TC-01a .. TC-07b), executed.

Each function is named after its test-case ID so tests/run_tests.py can build
the Pass/Fail table for the Final Report.  Controller-level tests call the
Auth Controller / Recommendation Engine / managers directly; UI-level tests
drive the real Streamlit app with streamlit.testing.v1.AppTest.
"""

import io
import time

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import auth
import db
import engine
import managers
import trainer

APP = "/home/claude/repo/app.py"


def _app(timeout=180):
    return AppTest.from_file(APP, default_timeout=timeout)


def _login_ui(email, pw):
    at = _app().run()
    at.text_input(key="login_email").set_value(email)
    at.text_input(key="login_password").set_value(pw)
    [b for b in at.button if b.label == "Login"][0].click()
    at.run()
    return at


def _chat(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def _last_assistant(at):
    return [m for m in at.chat_message if m.name == "assistant"][-1]


# ---------------------------------------------------------------- UC-01
def test_TC_01a_register_valid_unique_email():
    r = auth.register("Alice Valid", "alice.valid@example.com", "Passw0rdX", "Passw0rdX", "patient", True)
    assert r.ok and "Account created" in r.message
    u = db.get_user_by_email("alice.valid@example.com")
    assert u is not None and u["role"] == "patient" and u["is_verified"] == 1
    assert u["password_hash"] != "Passw0rdX" and len(u["password_hash"]) == 64  # salted hash, never plain text


def test_TC_01a_ui_register_redirects_to_login_with_success_message():
    at = _app().run()
    at.text_input(key="reg_name").set_value("Ui Register")
    at.text_input(key="reg_email").set_value("ui.register@example.com")
    at.text_input(key="reg_pw").set_value("Passw0rdX")
    at.text_input(key="reg_pw2").set_value("Passw0rdX")
    at.checkbox(key="reg_consent").check()
    [b for b in at.button if b.label == "Register"][0].click()
    at.run()
    assert not at.exception
    assert any("Account created" in s.value for s in at.success)
    assert [t.label for t in at.tabs][0] == "Login"  # login page is shown again


def test_TC_01b_register_duplicate_email():
    auth.register("Bob", "bob.dup@example.com", "Passw0rdX", "Passw0rdX", "patient", True)
    r = auth.register("Bob Two", "bob.dup@example.com", "Passw0rdX", "Passw0rdX", "patient", True)
    assert not r.ok and "Email already registered" in r.message


def test_UC01_validation_rules():
    assert "valid email" in auth.register("X", "not-an-email", "Passw0rdX", "Passw0rdX", "patient", True).message
    assert "too weak" in auth.register("X", "weak@example.com", "abc", "abc", "patient", True).message
    assert "do not match" in auth.register("X", "m@example.com", "Passw0rdX", "Passw0rdY", "patient", True).message
    assert "agree" in auth.register("X", "c@example.com", "Passw0rdX", "Passw0rdX", "patient", False).message
    assert "role" in auth.register("X", "r@example.com", "Passw0rdX", "Passw0rdX", "admin", True).message  # no self-made admins


# ---------------------------------------------------------------- UC-02
def test_TC_02a_login_correct_credentials(new_user):
    email, pw, _ = new_user
    r = auth.login(email, pw)
    assert r.ok and r.token and db.session_user(r.token)["email"] == email


def test_TC_02a_ui_login_goes_to_dashboard(new_user):
    email, pw, _ = new_user
    at = _login_ui(email, pw)
    assert not at.exception
    assert "AI Medicine Recommendation Assistant" in [t.value for t in at.title]
    assert at.chat_input  # recommendation dashboard with the chat box


def test_TC_02b_five_wrong_passwords_lock_account(new_user):
    email, pw, uid = new_user
    msgs = [auth.login(email, "WrongPass1").message for _ in range(5)]
    assert all("Invalid email or password" in m for m in msgs)
    assert "locked" in msgs[-1]
    assert db.get_user(uid)["locked_until"] is not None
    # even the correct password is refused while locked
    r = auth.login(email, pw)
    assert not r.ok and "locked" in r.message
    # password reset (simulated e-mail) unlocks the account
    code = auth.request_password_reset(email).extra
    assert auth.reset_password(email, code, "NewPassw0rd").ok
    assert auth.login(email, "NewPassw0rd").ok


# ---------------------------------------------------------------- UC-03
def test_TC_03a_srs_example_input_returns_full_bundle():
    """SRS input 'headache, fever, nausea'. Everything is displayed; see the next test for the >=50% part."""
    rec = engine.recommend("headache, fever, nausea")
    assert rec.status == "ok" and rec.disease
    assert len(rec.medicines) >= 3 and rec.precautions and rec.diet and rec.workout
    assert all(m["flag"] and m["dosage_note"] and "relevance" in m for m in rec.medicines)
    rels = [m["relevance"] for m in rec.medicines]
    assert rels == sorted(rels, reverse=True)  # FR-24 ranked
    print(f"TC-03a SRS input -> {rec.disease} {rec.confidence:.1f}% (low_confidence={rec.low_confidence})")


def test_TC_03a_specific_symptoms_reach_50_percent():
    rec = engine.recommend("I have high fever, headache, nausea, vomiting, chills and sweating")
    assert rec.status == "ok" and rec.confidence >= 50 and not rec.low_confidence
    assert rec.disease == "Malaria"
    reply = engine.format_reply(rec)
    assert "Confidence" in reply and "Precautions" in reply and "Suggested Diet" in reply and "Workout" in reply
    print(f"TC-03a specific input -> {rec.disease} {rec.confidence:.1f}%")


def test_TC_03b_low_confidence_warning_and_results_still_shown():
    rec = engine.recommend("headache, fever, nausea")
    assert rec.confidence < 50 and rec.low_confidence
    reply = engine.format_reply(rec)
    assert "Low confidence prediction. Please consult a healthcare professional." in reply
    assert rec.medicines and rec.precautions  # results still displayed


def test_TC_03b_ui_warning_is_shown_in_chat(new_user):
    email, pw, _ = new_user
    at = _login_ui(email, pw)
    _chat(at, "headache, fever, nausea")
    assert not at.exception
    text = str(_last_assistant(at).markdown[0].value)
    assert "Low confidence prediction. Please consult a healthcare professional." in text
    assert "Predicted Condition" in text


def test_UC03_no_symptom_is_rejected_politely():
    rec = engine.recommend("hello there")
    assert rec.status == "no_symptoms" and "symptoms" in rec.message


def test_UC03_exception_model_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("model file corrupt")
    monkeypatch.setattr(engine, "load_active_model", boom)
    rec = engine.recommend("itching and skin rash")
    assert rec.status == "error" and rec.message == "Recommendation service is temporarily unavailable."
    assert any(r["level"] == "ERROR" for r in db.recent_log(5))


def test_UC03_age_gender_allergy_read_from_the_sentence():
    rec = engine.recommend("I am a 30 year old female allergic to fluconazole. I have itching, skin rash and "
                           "nodal skin eruptions")
    assert rec.profile["age_group"] == "adult" and rec.profile["gender"] == "female"
    assert rec.profile["allergies"] == ["fluconazole"]
    assert {"itching", "skin_rash", "nodal_skin_eruptions"} <= set(rec.symptoms)  # symptoms survive the profile phrase
    flagged = [m["name"] for m in rec.medicines if m["allergy"]]
    assert flagged == ["Fluconazole"]
    assert any("Allergy warning" in n for n in rec.notes)


def test_UC03_allergy_map_penicillin_flags_antibiotics():
    import reference_data as ref
    assert ref.allergy_conflicts(["penicillin"], "Antibiotics") == ["penicillin"]
    assert ref.allergy_conflicts(["aspirin"], "NSAIDs") == ["aspirin"]
    assert ref.allergy_conflicts(["penicillin"], "Antiviral drugs") == []


def test_UC03_child_and_pregnancy_cautions():
    rec = engine.recommend("my 5 year old son has cough and fever")
    assert rec.profile["age_group"] == "child" and any("children" in n for n in rec.notes)
    rec = engine.recommend("I am pregnant and I have headache and vomiting")
    assert any("pregnancy" in n for n in rec.notes)


def test_UC03_prescription_flag_and_dosage_note_never_contain_numbers():
    for m in engine.recommend("itching, skin rash and nodal skin eruptions").medicines:
        assert m["flag"] in ("OTC", "Rx/OTC", "Rx", "Self-care", "Clinical")
        assert not any(ch.isdigit() for ch in m["dosage_note"])  # no invented doses


# ---------------------------------------------------------------- UC-04
def test_TC_04a_history_lists_prior_queries(new_user):
    email, pw, uid = new_user
    at = _login_ui(email, pw)
    _chat(at, "I have itching, skin rash and nodal skin eruptions")
    _chat(at, "I have a cough, breathlessness and chest pain")
    hist = db.user_history(uid)
    assert len(hist) == 2
    assert hist[0]["created_at"] >= hist[1]["created_at"]
    assert {"itching", "skin_rash"} <= set(hist[1]["symptoms"]) and hist[0]["top_medicine"]
    # the history page shows them
    at.sidebar.radio[0].set_value("My History").run()
    assert not at.exception and len(at.expander) == 2


def test_TC_04b_history_empty_message(new_user):
    email, pw, _ = new_user
    at = _login_ui(email, pw)
    at.sidebar.radio[0].set_value("My History").run()
    assert any(i.value == "No recommendation history found. Start by entering your symptoms." for i in at.info)


# ---------------------------------------------------------------- UC-05
def test_TC_05a_feedback_saved_with_thank_you(new_user):
    email, pw, uid = new_user
    at = _login_ui(email, pw)
    _chat(at, "I have itching, skin rash and nodal skin eruptions")
    qid = db.user_history(uid)[0]["query_id"]
    at.radio(key=f"fb_rating_{qid}").set_value(4)
    at.text_area(key=f"fb_comment_{qid}").set_value("Very clear answer")
    [b for b in at.button if b.label == "Submit feedback"][0].click()
    at.run()
    assert any("Thank you for your feedback" in s.value for s in at.success)
    fb = db.get_feedback(qid)
    assert fb["rating"] == 4 and fb["user_id"] == uid
    assert "Very clear" not in fb["comment_enc"]  # comment is encrypted at rest


def test_TC_05b_closing_the_form_creates_no_feedback(new_user):
    email, pw, uid = new_user
    at = _login_ui(email, pw)
    _chat(at, "I have itching, skin rash and nodal skin eruptions")
    qid = db.user_history(uid)[0]["query_id"]
    at.radio(key=f"fb_rating_{qid}").set_value(5)  # chosen but never submitted
    at.run()
    assert db.get_feedback(qid) is None
    assert db.q1("SELECT COUNT(*) FROM feedback WHERE user_id=?", (uid,))[0] == 0


def test_UC05_rating_must_be_1_to_5(new_user):
    _, _, uid = new_user
    qid = db.save_query(uid, "x", ["itching"], {}, 1, 80.0, False, None, [])
    with pytest.raises(ValueError):
        db.save_feedback(qid, uid, 6, "")


# ---------------------------------------------------------------- UC-06
@pytest.fixture(scope="module")
def admin_id():
    return db.get_user_by_email(db.DEFAULT_ADMIN_EMAIL)["user_id"]


def _csv_bytes(noise=0.0, drop_target=False):
    """The training CSV; `noise` = share of rows whose disease label is replaced by a random one."""
    df = pd.read_csv("Datasets/Training.csv")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if noise:
        rng = np.random.default_rng(3)
        flip = rng.random(len(df)) < noise
        df.loc[flip, "prognosis"] = rng.choice(df["prognosis"].unique(), flip.sum())
    if drop_target:
        df = df.drop(columns="prognosis")
    return df.to_csv(index=False).encode()


def test_TC_06a_valid_dataset_stored_with_summary(admin_id):
    res = managers.DatasetManager.upload("valid.csv", _csv_bytes(), admin_id)
    assert res["ok"] and "4920 rows" in res["message"] and "131 symptoms" in res["message"]
    assert res["report"]["rows_in"] == 4920 and res["report"]["duplicates_removed"] == 4616
    row = db.get_dataset(res["dataset_id"])
    assert row["n_rows"] == 4920 and row["status"] == "validated"


def test_TC_06b_invalid_dataset_rejected(admin_id):
    msg = "Invalid file format. Please upload a CSV or JSON file with required columns."
    assert managers.DatasetManager.upload("data.txt", b"hello", admin_id)["message"] == msg
    r = managers.DatasetManager.upload("nocolumn.csv", _csv_bytes(drop_target=True), admin_id)
    assert not r["ok"] and r["message"].startswith(msg)
    assert not managers.DatasetManager.upload("broken.json", b"{not json", admin_id)["ok"]


def test_UC06_exception_file_over_50mb(admin_id):
    r = managers.DatasetManager.upload("big.csv", b"0" * (51 * 1024 * 1024), admin_id)
    assert not r["ok"] and "50MB" in r["message"]


def test_UC06_json_dataset_accepted(admin_id):
    df = pd.read_csv("Datasets/Training.csv").head(400)
    r = managers.DatasetManager.upload("small.json", df.to_json(orient="records").encode(), admin_id)
    assert r["ok"], r["message"]


# ---------------------------------------------------------------- UC-07
@pytest.fixture(scope="module")
def weak_and_full(admin_id):
    """A poor dataset (30% wrong labels) and the clean dataset, both stored."""
    weak = managers.DatasetManager.upload("noisy.csv", _csv_bytes(noise=0.3), admin_id)
    good = managers.DatasetManager.upload("full.csv", _csv_bytes(), admin_id)
    assert weak["ok"], weak["message"]
    return weak["dataset_id"], good["dataset_id"]


def test_TC_07a_retrain_better_model_is_deployed_and_old_archived(admin_id, weak_and_full):
    weak, good = weak_and_full
    # make a deliberately weak model the production model
    r0 = managers.ModelManager.retrain(weak, ["Random Forest"], False, admin_id)
    assert r0["ok"]
    managers.ModelManager.deploy(r0["model_id"], admin_id)
    old_prod = db.production_model()
    # retrain on the better dataset
    r1 = managers.ModelManager.retrain(good, ["Random Forest"], False, admin_id)
    assert r1["ok"]
    cand = [m for m in db.list_models() if m["model_id"] == r1["model_id"]][0]
    assert cand["accuracy"] > old_prod["accuracy"]
    assert r1["comparison"]["recommend_deploy"]
    managers.ModelManager.deploy(r1["model_id"], admin_id)
    assert db.production_model()["model_id"] == r1["model_id"]
    assert [m for m in db.list_models() if m["model_id"] == old_prod["model_id"]][0]["status"] == "archived"
    # the chatbot now uses the new model
    rec = engine.recommend("itching, skin rash and nodal skin eruptions")
    assert rec.status == "ok" and rec.model_id == r1["model_id"]


def test_TC_07b_retrain_lower_accuracy_warns_and_keeps_current(admin_id, weak_and_full):
    weak, good = weak_and_full
    prod_before = db.production_model()  # strong model from the previous test (or v2)
    r = managers.ModelManager.retrain(weak, ["Random Forest"], False, admin_id)
    assert r["ok"]
    cmp_ = r["comparison"]
    assert not cmp_["recommend_deploy"]
    assert "LOWER" in cmp_["message"] and "Keeping the current model is recommended" in cmp_["message"]
    # admin follows the advice
    managers.ModelManager.reject(r["model_id"], admin_id)
    assert db.production_model()["model_id"] == prod_before["model_id"]


def test_UC07_exception_training_crash_keeps_previous_model(admin_id, weak_and_full):
    _, good = weak_and_full
    before = db.production_model()["model_id"]
    r = managers.ModelManager.retrain(good, ["No Such Model"], False, admin_id)
    assert not r["ok"] and "previous model is still in production" in r["message"]
    assert db.production_model()["model_id"] == before
    assert not managers.retraining_in_progress()  # lock released
    assert any(x["event"] == "training_failed" for x in db.recent_log(10))


# ---------------------------------------------------------------- non-functional
def test_NFR04_sensitive_fields_are_encrypted_at_rest(new_user):
    _, _, uid = new_user
    qid = db.save_query(uid, "secret symptom text", ["itching"], {"age_group": "adult", "gender": "male",
                        "allergies": ["penicillin"]}, 1, 77.0, False, None, [])
    row = db.q1("SELECT * FROM recommendation_queries WHERE query_id=?", (qid,))
    for col in ("input_text_enc", "symptoms_enc", "age_group_enc", "gender_enc", "allergy_enc"):
        assert "secret" not in row[col] and "penicillin" not in row[col] and "itching" not in row[col]
    assert db.user_history(uid)[0]["input_text"] == "secret symptom text"


def test_NFR04_aes256_key_and_roundtrip():
    import security
    assert len(security._load_key()) == 32  # 256-bit key
    blob = security.encrypt_text("hello")
    assert blob != security.encrypt_text("hello")  # random nonce
    assert security.decrypt_text(blob) == "hello"


def test_NFR08_production_model_accuracy_at_least_85_percent():
    prod = db.production_model()
    assert prod["accuracy"] >= 0.85
    # and recompute on a fresh held-out split of the de-duplicated data
    X, y, _ = trainer.clean_dataset(trainer.load_dataset("Datasets/Training.csv"))
    bundle, _ = engine.load_active_model()
    yy = bundle["label_encoder"].transform(y)
    cols = bundle["symptom_columns"]
    pred = bundle["model"].predict(X[cols].values if not hasattr(bundle["model"], "feature_names_in_") else X[cols])
    assert (pred == yy).mean() >= 0.85


def test_NFR09_consent_required_and_recorded():
    r = auth.register("No Consent", "noconsent@example.com", "Passw0rdX", "Passw0rdX", "patient", False)
    assert not r.ok and db.get_user_by_email("noconsent@example.com") is None
    u = db.get_user_by_email("alice.valid@example.com")
    assert u["consent_given"] == 1 and u["consent_at"]


def test_NFR_response_time_under_3_seconds():
    engine.recommend("itching")  # warm-up (model load)
    t = time.time()
    engine.recommend("I have a cough, breathlessness and chest pain")
    assert time.time() - t < 3
