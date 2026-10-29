"""
Medicine Recommendation System - AI Chatbot Assistant (final application)
CS619 Final Year Project | Group ID: S26PROJECTFEF30

Run with:  streamlit run app.py

Three-tier layout (Design Document section 4):
  presentation  : this file (Streamlit)
  application   : auth.py, engine.py, managers.py, trainer.py
  data          : db.py (SQLite), Models/ (.pkl files), data/datasets/ (uploads)

The system is a chatbot: the only way to enter symptoms is a free-text sentence.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
import db
import engine
import managers
import trainer

st.set_page_config(page_title="Medicine Recommendation Assistant", page_icon="💊", layout="wide")


@st.cache_resource
def _bootstrap():
    db.init_db()
    return True


_bootstrap()

WELCOME = ("Hi! Tell me what symptoms you are experiencing, in your own words, and I will try to help figure "
           "out what might be going on. You can also mention your age, gender or allergies in the same message, "
           "for example: *\"I am 30, female, allergic to penicillin, with a high fever, chills and headache.\"*")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def current_user():
    token = st.session_state.get("token")
    user = db.session_user(token)
    if token and user is None:
        st.session_state.pop("token", None)
    return user


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


@st.cache_data
def load_training_preview():
    return pd.read_csv("Datasets/Training.csv")


# ---------------------------------------------------------------------------
# Login / Register / Forgot password   (UC-01, UC-02)
# ---------------------------------------------------------------------------
def page_auth():
    st.title("💊 Medicine Recommendation Assistant")
    st.caption("AI chatbot that suggests a likely condition and medicines from the symptoms you describe.")
    if st.session_state.pop("just_registered", None):
        st.success(st.session_state.pop("register_msg", "Account created. Please log in."))

    tab_login, tab_reg, tab_forgot = st.tabs(["Login", "Register", "Forgot password"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Login", type="primary"):
                res = auth.login(email, password)
                if res.ok:
                    st.session_state.token = res.token
                    st.session_state.pop("chat_history", None)
                    st.session_state.pop("profile", None)
                    st.rerun()
                else:
                    st.error(res.message)

    with tab_reg:
        with st.form("register_form"):
            name = st.text_input("Full name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            c1, c2 = st.columns(2)
            pw = c1.text_input("Password", type="password", key="reg_pw",
                               help="At least 8 characters with an uppercase letter, a lowercase letter and a digit.")
            pw2 = c2.text_input("Confirm password", type="password", key="reg_pw2")
            role = st.selectbox("I am a", ["patient", "provider", "pharmacist"], key="reg_role",
                                format_func=lambda r: {"patient": "Patient", "provider": "Healthcare Provider",
                                                       "pharmacist": "Pharmacist"}[r])
            consent = st.checkbox("I agree that my symptom queries and feedback are stored (encrypted) so I can "
                                  "view my history.", key="reg_consent")
            if st.form_submit_button("Register", type="primary"):
                res = auth.register(name, email, pw, pw2, role, consent)
                if res.ok:
                    st.session_state.just_registered = True
                    st.session_state.register_msg = res.message
                    st.rerun()
                else:
                    st.error(res.message)

    with tab_forgot:
        st.caption("No mail server is configured in this project, so the reset code is shown here (simulated e-mail).")
        with st.form("reset_request_form"):
            email = st.text_input("Registered email", key="fp_email")
            if st.form_submit_button("Send reset code"):
                res = auth.request_password_reset(email)
                st.info(res.message)
                if res.extra:
                    st.code(f"Reset code: {res.extra}")
        with st.form("reset_form"):
            email2 = st.text_input("Email", key="fp_email2")
            code = st.text_input("Reset code", key="fp_code")
            newpw = st.text_input("New password", type="password", key="fp_newpw")
            if st.form_submit_button("Set new password"):
                res = auth.reset_password(email2, code, newpw)
                (st.success if res.ok else st.error)(res.message)


# ---------------------------------------------------------------------------
# Chatbot   (UC-03 and UC-05)
# ---------------------------------------------------------------------------
def feedback_widget(query_id: int, user_id: int):
    """UC-05: 1-5 stars + optional comment. Nothing is saved unless Submit is pressed."""
    existing = db.get_feedback(query_id)
    label = f"⭐ Rated {existing['rating']}/5 - change rating" if existing else "⭐ Rate this recommendation"
    with st.expander(label):
        with st.form(f"fb_form_{query_id}"):
            rating = st.radio("How helpful was this recommendation?", [1, 2, 3, 4, 5], horizontal=True,
                              format_func=lambda n: "★" * n, index=None, key=f"fb_rating_{query_id}")
            comment = st.text_area("Comment (optional)", key=f"fb_comment_{query_id}")
            if st.form_submit_button("Submit feedback"):
                if rating is None:
                    st.warning("Please choose a rating from 1 to 5 stars.")
                else:
                    try:
                        db.save_feedback(query_id, user_id, rating, comment)
                        st.success("Thank you for your feedback!")
                    except Exception as exc:
                        db.log_event("feedback_failed", repr(exc), level="ERROR", user_id=user_id)
                        st.error("Your feedback could not be saved. Please try again.")


def render_chat_message(msg: dict, user):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("top3"):
            with st.expander("Other possible matches"):
                for name, conf in msg["top3"]:
                    st.write(f"- {name}: {conf:.1f}%")
        if msg.get("query_id"):
            st.caption("Relevance = how strongly a treatment is linked to the most likely conditions, "
                       "weighted by each condition's probability.")
            feedback_widget(msg["query_id"], user["user_id"])


def page_chat(user):
    st.title("AI Medicine Recommendation Assistant")
    st.caption("Describe how you feel in your own words, for example \"I have itching, a skin rash, and some "
               "nodal skin eruptions.\" The assistant picks out the symptoms it recognises and suggests what "
               "might be going on.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": WELCOME}]
    st.session_state.setdefault("profile", None)

    for msg in st.session_state.chat_history:
        render_chat_message(msg, user)

    text = st.chat_input("Describe your symptoms...")
    if text:
        st.session_state.chat_history.append({"role": "user", "content": text})
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant"):
            with st.spinner("Analysing your symptoms..."):
                rec = engine.recommend(text, st.session_state.profile)
        st.session_state.profile = rec.profile or st.session_state.profile
        reply = {"role": "assistant", "content": engine.format_reply(rec)}
        if rec.status == "ok":
            try:
                rec.query_id = db.save_query(
                    user["user_id"], text, list(rec.symptoms), rec.profile, rec.disease_id, rec.confidence,
                    rec.low_confidence, rec.model_id, rec.medicines)
                reply["query_id"] = rec.query_id
            except Exception as exc:  # history could not be saved, results are still shown
                db.log_event("save_query_failed", repr(exc), level="ERROR", user_id=user["user_id"])
                reply["content"] += "\n\n*Your history could not be saved this time.*"
            reply["top3"] = rec.top3
        st.session_state.chat_history.append(reply)
        st.rerun()

    if len(st.session_state.chat_history) > 1:
        if st.button("Start new conversation"):
            st.session_state.chat_history = [{"role": "assistant", "content": WELCOME}]
            st.session_state.profile = None
            st.rerun()


# ---------------------------------------------------------------------------
# History   (UC-04)
# ---------------------------------------------------------------------------
def page_history(user):
    st.title("My History")
    try:
        items = db.user_history(user["user_id"])
    except Exception as exc:
        db.log_event("history_failed", repr(exc), level="ERROR", user_id=user["user_id"])
        st.error("The history could not be loaded. Please refresh the page.")
        return
    if not items:
        st.info("No recommendation history found. Start by entering your symptoms.")
        return
    st.caption(f"{len(items)} previous queries, newest first.")
    for it in items:
        sym = ", ".join(s.replace("_", " ") for s in it["symptoms"])
        title = f"{it['created_at'].replace('T', ' ')} - {it['disease']} ({it['confidence']:.0f}%)"
        with st.expander(title):
            st.write(f"**You wrote:** {it['input_text']}")
            st.write(f"**Symptoms detected:** {sym}")
            if it["age_group"] or it["gender"] or it["allergy"]:
                st.write(f"**Noted:** {it['age_group'] or '-'} / {it['gender'] or '-'} / allergies: {it['allergy'] or '-'}")
            st.write(f"**Top recommendation:** {it['top_medicine']}")
            if it["low_confidence"]:
                st.warning(engine.LOW_CONFIDENCE_MSG)
            meds = db.query_medicines(it["query_id"])
            st.dataframe(pd.DataFrame([{
                "#": m["rank_order"], "Medicine / treatment": m["name"] + (" ⚠️" if m["allergy_flag"] else ""),
                "Relevance": f"{m['relevance_score']:.0f}%", "Prescription": m["prescription_flag"]} for m in meds]),
                hide_index=True, width="stretch")
            st.write(f"**Your rating:** {'★' * it['rating'] if it['rating'] else 'not rated'}")


# ---------------------------------------------------------------------------
# Reference pages (carried over from the prototype)
# ---------------------------------------------------------------------------
def page_dataset_preview():
    training = load_training_preview()
    prod = db.production_model()
    st.title("Dataset Preview")
    st.write(f"Training set: **{training.shape[0]} rows × {training.shape[1]} columns** "
             f"({training.shape[1] - 1} binary symptom features, {training['prognosis'].nunique()} diseases).")
    st.subheader("Sample rows")
    st.dataframe(training.head(10))
    st.subheader("Diseases covered")
    counts = training["prognosis"].value_counts().reset_index()
    counts.columns = ["Disease", "Rows"]
    st.dataframe(counts, height=300)
    st.subheader("Reference tables in the database")
    rows = {t: db.q1(f"SELECT COUNT(*) FROM {t}")[0] for t in
            ["diseases", "symptoms", "disease_symptom_map", "medicines", "disease_medicine_map", "care_guidance"]}
    st.dataframe(pd.DataFrame({"Table": rows.keys(), "Rows": rows.values()}), hide_index=True)


def page_model_evaluation():
    st.title("Model Evaluation")
    prod = db.production_model()
    if prod is None:
        st.warning("No production model is registered.")
        return
    st.write(f"Production model: **{prod['algorithm']}** (version {prod['version']}), "
             f"deployed {str(prod['deployed_at'] or '-')[:16]}.")
    c = st.columns(5)
    c[0].metric("Test accuracy", pct(prod["accuracy"]))
    c[1].metric("Precision", pct(prod["precision_w"]))
    c[2].metric("Recall", pct(prod["recall_w"]))
    c[3].metric("F1", pct(prod["f1_w"]))
    c[4].metric("Accuracy, 3 symptoms", pct(prod["acc_3_symptoms"]))
    st.caption("Metrics come from the held-out 20% test split of the de-duplicated data (no train/test overlap). "
               "Near-perfect accuracy with all symptoms is expected: the dataset encodes each disease as a fixed "
               "combination of symptoms. The more informative number is the accuracy when only 3 symptoms are given.")
    csv = Path("Models/model_comparison.csv")
    if csv.exists():
        st.subheader("Model comparison (Notebooks/model_comparison.ipynb)")
        df = pd.read_csv(csv, index_col=0)
        show = [c for c in ["Accuracy", "Precision", "Recall", "F1", "Acc@2 symptoms", "Acc@3 symptoms", "Acc@4 symptoms",
                            "CV acc (mean)", "CV acc@3 (mean)", "Overfit gap", "Selected"] if c in df.columns]
        st.dataframe(df[show].style.format({c: "{:.3f}" for c in show if c != "Selected"}, na_rep="-"))
        fig = Path("Notebooks/figures/model_comparison.png")
        if fig.exists():
            st.image(str(fig))
    st.subheader("Feature importance (top 15 symptoms)")
    try:
        bundle, _ = engine.load_active_model()
        if hasattr(bundle["model"], "feature_importances_"):
            imp = pd.Series(bundle["model"].feature_importances_, index=bundle["symptom_columns"])
            st.bar_chart(imp.sort_values(ascending=False).head(15))
        else:
            st.info("The active model does not expose feature importances.")
    except Exception as exc:
        st.error(f"{engine.UNAVAILABLE_MSG} ({exc})")


# ---------------------------------------------------------------------------
# Admin: datasets and models   (UC-06, UC-07)
# ---------------------------------------------------------------------------
def tab_datasets(admin):
    st.subheader("Upload a dataset")
    st.caption("CSV or JSON, maximum 50 MB. Required: a `prognosis` column (disease) and one 0/1 column per symptom.")
    up = st.file_uploader("Choose a file", key="ds_upload")
    c1, c2 = st.columns([1, 3])
    if c1.button("Validate and store", type="primary", disabled=up is None):
        res = managers.DatasetManager.upload(up.name, up.getvalue(), admin["user_id"])
        (st.success if res["ok"] else st.error)(res["message"])
        if res["ok"]:
            st.json(res["report"], expanded=False)
    if c2.button("Register the built-in training dataset (Datasets/Training.csv)"):
        content = Path("Datasets/Training.csv").read_bytes()
        res = managers.DatasetManager.upload("Training.csv", content, admin["user_id"])
        (st.success if res["ok"] else st.error)(res["message"])
    st.subheader("Stored datasets")
    rows = db.list_datasets()
    if rows:
        st.dataframe(pd.DataFrame([dict(r) for r in rows]).drop(columns=["stored_path"]), hide_index=True)
    else:
        st.info("No dataset has been stored yet.")


def comparison_table(cmp_):
    df = pd.DataFrame(cmp_["rows"])
    if df.empty:
        return
    for c in ["Current production", "New candidate"]:
        df[c] = df[c].map(lambda v: "-" if v is None else f"{v * 100:.1f}%")
    df["Change"] = df["Change"].map(lambda v: "-" if v is None else f"{v * 100:+.1f} pts")
    st.dataframe(df, hide_index=True, width="stretch")


def tab_retrain(admin):
    datasets = db.list_datasets()
    if not datasets:
        st.info("Upload or register a dataset first (tab 'Datasets').")
    else:
        st.subheader("Train new model(s)")
        opts = {f"#{d['dataset_id']} - {d['filename']} ({d['n_rows']} rows)": d["dataset_id"] for d in datasets}
        label = st.selectbox("Dataset", list(opts))
        models = st.multiselect("Model types to train and compare", trainer.MODEL_NAMES,
                                default=["Random Forest", "Decision Tree", "SVM"])
        tune = st.checkbox("Tune hyper-parameters with GridSearchCV (slower, a few minutes)", value=False)
        if managers.retraining_in_progress():
            st.warning("A retraining job is already running.")
        if st.button("Train and evaluate", type="primary", disabled=not models):
            with st.status("Training...", expanded=True) as status:
                res = managers.ModelManager.retrain(opts[label], models, tune, admin["user_id"], progress=st.write)
                status.update(label=res["message"], state="complete" if res["ok"] else "error")
            if res["ok"]:
                st.dataframe(res["table"].drop(columns=["Best params"]).style.format(
                    {c: "{:.3f}" for c in res["table"].columns if c not in ("Selected", "Best params", "Train sec")}),
                    width="stretch")
            else:
                st.error(res["message"])

    cands = [m for m in db.list_models() if m["status"] == "candidate"]
    if cands:
        st.subheader("Candidate models waiting for a decision")
        prod = db.production_model()
        for m in cands:
            cmp_ = managers.compare_with_production(m, prod)
            with st.container(border=True):
                st.markdown(f"**{m['version']} - {m['algorithm']}** (trained {str(m['trained_at'])[:16]})")
                comparison_table(cmp_)
                (st.success if cmp_["recommend_deploy"] else st.warning)(cmp_["message"])
                b1, b2, _ = st.columns([1, 1, 3])
                if b1.button("Deploy model", key=f"deploy_{m['model_id']}", type="primary"):
                    st.success(managers.ModelManager.deploy(m["model_id"], admin["user_id"]))
                    st.rerun()
                if b2.button("Keep current model", key=f"keep_{m['model_id']}"):
                    st.info(managers.ModelManager.reject(m["model_id"], admin["user_id"]))
                    st.rerun()


def tab_registry(admin):
    st.subheader("Model registry")
    rows = db.list_models()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        st.info("No models registered.")
        return
    view = df[["model_id", "version", "algorithm", "status", "accuracy", "f1_w", "acc_3_symptoms", "cv_accuracy",
               "trained_at", "deployed_at"]]
    st.dataframe(view, hide_index=True, width="stretch")
    archived = [r for r in rows if r["status"] == "archived" and Path(r["file_path"]).exists()]
    if archived:
        pick = st.selectbox("Roll back to an archived model", [f"{r['version']} - {r['algorithm']}" for r in archived])
        if st.button("Roll back"):
            mid = archived[[f"{r['version']} - {r['algorithm']}" for r in archived].index(pick)]["model_id"]
            st.success(managers.ModelManager.rollback(mid, admin["user_id"]))
            st.rerun()


def page_admin_models(admin):
    st.title("Admin Panel - Dataset & Model Management")
    t1, t2, t3 = st.tabs(["Datasets", "Retrain and deploy", "Model registry"])
    with t1:
        tab_datasets(admin)
    with t2:
        tab_retrain(admin)
    with t3:
        tab_registry(admin)


def page_admin_users(admin):
    st.title("User Management")
    users = db.list_users()
    st.dataframe(pd.DataFrame([dict(u) for u in users]), hide_index=True, width="stretch")
    options = {f"{u['full_name']} <{u['email']}>": u for u in users}
    who = options[st.selectbox("Select a user", list(options))]
    c1, c2, c3 = st.columns(3)
    new_role = c1.selectbox("Role", list(db.ROLES), index=list(db.ROLES).index(who["role"]))
    if c1.button("Change role"):
        if who["user_id"] == admin["user_id"]:
            st.error("You cannot change your own role.")
        else:
            db.execute("UPDATE users SET role=? WHERE user_id=?", (new_role, who["user_id"]))
            db.log_event("role_changed", f"user {who['user_id']} -> {new_role}", user_id=admin["user_id"])
            st.success("Role updated.")
            st.rerun()
    if c2.button("Deactivate" if who["is_active"] else "Activate"):
        if who["user_id"] == admin["user_id"]:
            st.error("You cannot deactivate your own account.")
        else:
            db.execute("UPDATE users SET is_active=? WHERE user_id=?", (0 if who["is_active"] else 1, who["user_id"]))
            db.log_event("user_active_toggled", f"user {who['user_id']}", user_id=admin["user_id"])
            st.rerun()
    if c3.button("Unlock account"):
        db.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE user_id=?", (who["user_id"],))
        db.log_event("user_unlocked", f"user {who['user_id']}", user_id=admin["user_id"])
        st.success("Account unlocked.")
        st.rerun()


def page_admin_reports(admin):
    st.title("Reports")
    s = db.report_summary()
    c = st.columns(6)
    c[0].metric("Users", s["users"])
    c[1].metric("Queries", s["queries"])
    c[2].metric("Low-confidence", s["low_confidence"])
    c[3].metric("Feedback entries", s["feedback_count"])
    c[4].metric("Average rating", s["avg_rating"] if s["avg_rating"] is not None else "-")
    c[5].metric("Average confidence", f"{s['avg_confidence']}%" if s["avg_confidence"] is not None else "-")
    left, right = st.columns(2)
    with left:
        st.subheader("Most frequent predicted conditions")
        td = db.top_diseases()
        if td:
            df = pd.DataFrame([dict(r) for r in td])
            st.bar_chart(df.set_index("disease")["queries"])
        else:
            st.info("No queries yet.")
    with right:
        st.subheader("Feedback ratings")
        rd = db.rating_distribution()
        if rd:
            df = pd.DataFrame([dict(r) for r in rd]).set_index("rating")
            st.bar_chart(df["n"])
        else:
            st.info("No feedback yet.")
    st.subheader("System log (latest 100 events)")
    st.dataframe(pd.DataFrame([dict(r) for r in db.recent_log()]), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
user = current_user()
if user is None:
    page_auth()
    st.stop()

PAGES = ["AI Chatbot", "My History", "Dataset Preview", "Model Evaluation"]
if user["role"] == "admin":
    PAGES += ["Admin: Datasets & Models", "Admin: Users", "Admin: Reports"]

st.sidebar.title("💊 Medicine Recommendation Assistant")
st.sidebar.caption("CS619 Final Project - Group S26PROJECTFEF30")
st.sidebar.write(f"Signed in as **{user['full_name']}** ({user['role']})")
page = st.sidebar.radio("Navigate", PAGES)
if st.sidebar.button("Log out"):
    auth.logout(st.session_state.get("token"))
    for k in ("token", "chat_history", "profile"):
        st.session_state.pop(k, None)
    st.rerun()

if page == "AI Chatbot":
    page_chat(user)
elif page == "My History":
    page_history(user)
elif page == "Dataset Preview":
    page_dataset_preview()
elif page == "Model Evaluation":
    page_model_evaluation()
elif page == "Admin: Datasets & Models":
    page_admin_models(user)
elif page == "Admin: Users":
    page_admin_users(user)
elif page == "Admin: Reports":
    page_admin_reports(user)
