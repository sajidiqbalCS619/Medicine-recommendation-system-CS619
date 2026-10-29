"""
Generates every diagram used in the Final Report and the Final Presentation.

    python docs/diagrams/make_diagrams.py

Outputs PNG files next to this script.  The ERD and the database diagram are
read from the real SQLite schema in db.py, so they always match the code.
Requires: graphviz (dot) and matplotlib.
"""
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

TEAL, DARK, LIGHT, AMBER = "#0F766E", "#134E4A", "#E6F4F1", "#F59E0B"


def dot(name, source):
    f = HERE / f"{name}.dot"
    f.write_text(source, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=170", str(f), "-o", str(HERE / f"{name}.png")], check=True)
    f.unlink()
    print("made", name)


FONT = 'fontname="Helvetica"'

# ---------------------------------------------------------------- process model
dot("process_model", f"""
digraph G {{
  rankdir=LR; nodesep=0.35; ranksep=0.45; {FONT}; bgcolor="white";
  node [shape=box, style="rounded,filled", fillcolor="{LIGHT}", color="{TEAL}", {FONT}, fontsize=12, margin="0.18,0.1"];
  edge [color="{DARK}", {FONT}, fontsize=10];
  subgraph cluster_w {{ label="VU Process Model - Waterfall backbone"; style="rounded"; color="{TEAL}"; fontsize=14; {FONT};
    p1 [label="1. Requirements\\nAnalysis\\n(SRS)"];
    p2 [label="2. System\\nDesign\\n(Design Doc)"];
    p3 [label="3. Implementation"];
    p4 [label="4. Testing"];
    p5 [label="5. Deployment"];
    p6 [label="6. Maintenance"];
    p1 -> p2 -> p3 -> p4 -> p5 -> p6;
  }}
  subgraph cluster_s {{ label="Spiral cycles inside Implementation (risk-driven model development)"; style="rounded,dashed"; color="{AMBER}"; fontsize=13; {FONT};
    node [fillcolor="#FEF3C7", color="{AMBER}"];
    s1 [label="Spiral 1\\nPrototype &\\nbaseline model"];
    s2 [label="Spiral 2\\nData enrichment &\\nmodel comparison"];
    s3 [label="Spiral 3\\nHyper-parameter\\ntuning"];
    s4 [label="Spiral 4\\nIntegration &\\nfinal validation"];
    s1 -> s2 -> s3 -> s4;
    s4 -> s1 [label="risk analysis,\\nnext iteration", style=dashed, constraint=false];
  }}
  p3 -> s1 [style=dashed, color="{AMBER}"];
  s4 -> p4 [style=dashed, color="{AMBER}"];
  p6 -> p1 [label="change requests /\\nmodel retraining", style=dotted, constraint=false];
}}
""")

# ---------------------------------------------------------------- use case diagram
dot("use_case", f"""
digraph G {{
  rankdir=LR; nodesep=0.3; ranksep=1.2; splines=line; {FONT}; bgcolor="white";
  node [{FONT}, fontsize=12];
  edge [color="{DARK}", arrowhead=none];
  user [shape=box, style="filled", fillcolor="{LIGHT}", color="{TEAL}", label="Patient /\\nHealthcare Provider /\\nPharmacist\\n(actor)"];
  admin [shape=box, style="filled", fillcolor="#FEF3C7", color="{AMBER}", label="System\\nAdministrator\\n(actor)"];
  subgraph cluster_sys {{ label="Medicine Recommendation System (AI chatbot)"; style="rounded"; color="{TEAL}"; fontsize=14;
    node [shape=ellipse, style=filled, fillcolor="white", color="{TEAL}"];
    uc1 [label="UC-01\\nRegister"]; uc2 [label="UC-02\\nLogin"];
    uc3 [label="UC-03\\nDescribe symptoms &\\nget recommendations"];
    uc4 [label="UC-04\\nView history"]; uc5 [label="UC-05\\nGive feedback"];
    uc6 [label="UC-06\\nManage datasets"]; uc7 [label="UC-07\\nRetrain & deploy model"];
  }}
  user -> uc1; user -> uc2; user -> uc3; user -> uc4; user -> uc5;
  admin -> uc2; admin -> uc6; admin -> uc7;
  uc3 -> uc2 [style=dashed, arrowhead=vee, label="<<include>>", fontsize=10];
  uc4 -> uc2 [style=dashed, arrowhead=vee, label="<<include>>", fontsize=10];
  uc5 -> uc3 [style=dashed, arrowhead=vee, label="<<extend>>", fontsize=10];
  uc7 -> uc6 [style=dashed, arrowhead=vee, label="<<include>>", fontsize=10];
}}
""")

# ---------------------------------------------------------------- architecture
dot("architecture", f"""
digraph G {{
  rankdir=TB; nodesep=0.35; ranksep=0.6; {FONT}; bgcolor="white"; compound=true;
  node [shape=box, style="rounded,filled", fillcolor="white", color="{TEAL}", {FONT}, fontsize=12, margin="0.2,0.12"];
  edge [color="{DARK}", {FONT}, fontsize=10];
  user [shape=box, style="filled", fillcolor="{LIGHT}", label="Users: Patient / Provider / Pharmacist / Administrator\\n(Chrome, Firefox, Edge - desktop or mobile)"];
  subgraph cluster_p {{ label="Presentation tier"; style="rounded,filled"; fillcolor="#F0FDFA"; color="{TEAL}"; fontsize=14;
    ui [label="Streamlit web application (app.py)\\nLogin/Register | AI Chatbot | My History | Model Evaluation | Admin panel"]; }}
  subgraph cluster_a {{ label="Application / business-logic tier"; style="rounded,filled"; fillcolor="#FFFBEB"; color="{AMBER}"; fontsize=14;
    auth [label="Auth Controller\\n(auth.py)"]; eng [label="Recommendation Engine\\n(engine.py)\\nNLP + predict + rank"];
    mm [label="Model Manager\\n(managers.py, trainer.py)"]; dm [label="Dataset Manager\\n(managers.py)"]; sec [label="Security\\n(security.py)\\nPBKDF2, AES-256-GCM"]; }}
  subgraph cluster_d {{ label="Data tier"; style="rounded,filled"; fillcolor="#F0F9FF"; color="#0369A1"; fontsize=14;
    sql [shape=cylinder, label="SQLite database\\n(data/mrs.db)\\n15 tables"]; ms [shape=folder, label="Model store\\n(Models/versions/*.pkl)"]; ds [shape=folder, label="Dataset store\\n(Datasets/, data/datasets/)"]; }}
  user -> ui [label="HTTPS"];
  ui -> auth; ui -> eng; ui -> mm; ui -> dm;
  auth -> sec; eng -> sec; auth -> sql; eng -> sql; eng -> ms [label="load active model"];
  mm -> ms [label="save .pkl"]; mm -> sql [label="register/deploy"]; dm -> ds; dm -> sql; mm -> ds [label="read"];
}}
""")

# ---------------------------------------------------------------- team structure
dot("team_structure", f"""
digraph G {{
  rankdir=TB; nodesep=0.4; ranksep=0.5; {FONT}; bgcolor="white";
  node [shape=box, style="rounded,filled", fillcolor="{LIGHT}", color="{TEAL}", {FONT}, fontsize=12, margin="0.2,0.12"];
  edge [color="{DARK}", arrowhead=none];
  sup [label="Supervisor\\nDr. Mushtaq Hussain\\nmushtaq.hussain@vu.edu.pk", fillcolor="#FEF3C7", color="{AMBER}"];
  team [label="Project Group S26PROJECTFEF30\\n(Team ID: themushtaq48)"];
  m1 [label="Member 1\\nBC250226691"]; m2 [label="Member 2\\nBC210406922"];
  r1 [shape=note, fillcolor="white", label="Shared responsibilities\\n- requirements & design documents\\n- data pre-processing & model training\\n- recommendation engine & chatbot\\n- database, security & admin panel\\n- testing, report & presentation"];
  sup -> team -> m1; team -> m2; m1 -> r1 [style=dashed]; m2 -> r1 [style=dashed];
}}
""")

# ---------------------------------------------------------------- class diagram
def rec(name, attrs, methods):
    a = "\\l".join(attrs) + "\\l" if attrs else ""
    m = "\\l".join(methods) + "\\l" if methods else ""
    return f'{name} [label="{{{name}|{a}|{m}}}"];'


dot("class_diagram", f"""
digraph G {{
  rankdir=BT; nodesep=0.45; ranksep=0.7; {FONT}; bgcolor="white";
  node [shape=record, style=filled, fillcolor="{LIGHT}", color="{TEAL}", {FONT}, fontsize=10];
  edge [color="{DARK}", {FONT}, fontsize=9];
  {rec("User", ["- user_id : int", "- full_name : str", "- email : str", "- password_hash : str", "- role : Role", "- failed_attempts : int", "- locked_until : datetime"], ["+ register()", "+ login()", "+ logout()", "+ resetPassword()"])}
  {rec("Administrator", [], ["+ uploadDataset()", "+ retrainModel()", "+ deployModel()", "+ manageUsers()", "+ viewReports()"])}
  {rec("AuthController", [], ["+ register(name, email, pw, role, consent)", "+ login(email, pw) : Session", "+ requestPasswordReset()", "+ resetPassword()"])}
  {rec("RecommendationEngine", ["- LOW_CONFIDENCE = 50 %"], ["+ extractSymptoms(text) : dict", "+ extractProfile(text) : Profile", "+ recommend(text) : Recommendation", "+ rankMedicines(top3) : list", "+ formatReply()"])}
  {rec("MLModel", ["- version : str", "- algorithm : str", "- accuracy, precision, recall, f1", "- status : candidate / production / archived", "- filePath : str"], ["+ train()", "+ tune()", "+ evaluate()", "+ save() / load()", "+ predictProba()"])}
  {rec("ModelManager", [], ["+ retrain(dataset, models, tune)", "+ compareWithProduction()", "+ deploy(model)", "+ rollback(model)"])}
  {rec("DatasetManager", [], ["+ upload(file)", "+ validate()", "+ clean() / deduplicate()", "+ encode()"])}
  {rec("Disease", ["- disease_id : int", "- name : str", "- description : str"], [])}
  {rec("Symptom", ["- symptom_id : int", "- name : str", "- severity : int"], [])}
  {rec("Medicine", ["- medicine_id : int", "- name : str", "- prescription_flag : Rx / OTC / ...", "- dosage_note : str"], [])}
  {rec("RecommendationQuery", ["- query_id : int", "- created_at : datetime", "- input_text (AES-256)", "- confidence : float", "- low_confidence : bool"], ["+ save()", "+ details()"])}
  {rec("FeedbackModel", ["- rating : 1..5", "- comment (AES-256)"], ["+ submit()"])}
  {rec("Dataset", ["- filename, rows, columns", "- symptoms, diseases", "- duplicates_removed"], [])}
  Administrator -> User [arrowhead=empty, label="inherits"];
  AuthController -> User [arrowhead=vee, style=dashed, label="manages"];
  RecommendationEngine -> MLModel [arrowhead=vee, label="uses active"];
  RecommendationEngine -> Disease [arrowhead=vee];
  RecommendationEngine -> Medicine [arrowhead=vee];
  RecommendationEngine -> RecommendationQuery [arrowhead=vee, label="logs"];
  RecommendationQuery -> User [arrowhead=vee, label="1..*  belongs to"];
  RecommendationQuery -> Disease [arrowhead=vee, label="predicts"];
  RecommendationQuery -> Medicine [arrowhead=vee, label="ranked *"];
  FeedbackModel -> RecommendationQuery [arrowhead=vee, label="0..1  rates"];
  Disease -> Symptom [arrowhead=vee, label="* has *"];
  Disease -> Medicine [arrowhead=vee, label="* treated by * (relevance)"];
  ModelManager -> MLModel [arrowhead=diamond, label="manages"];
  ModelManager -> DatasetManager [arrowhead=vee, style=dashed, label="reads"];
  DatasetManager -> Dataset [arrowhead=vee, label="stores"];
  Administrator -> ModelManager [arrowhead=vee, style=dashed];
  Administrator -> DatasetManager [arrowhead=vee, style=dashed];
}}
""")

# ---------------------------------------------------------------- ERD + database diagram from the real schema
import db  # noqa: E402

conn = sqlite3.connect(":memory:")
conn.executescript(db.SCHEMA)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
fks = {}
cols = {}
for t in tables:
    cols[t] = list(conn.execute(f"PRAGMA table_info({t})"))
    for fk in conn.execute(f"PRAGMA foreign_key_list({t})"):
        fks.setdefault(t, []).append((fk[3], fk[2], fk[4]))
    fks.setdefault(t, [])


def table_node(t):
    fk_cols = {c for c, _, _ in fks[t]}
    rows = []
    for _, name, typ, notnull, _, pk in cols[t]:
        tag = "PK" if pk else ("FK" if name in fk_cols else "")
        bg = "#FEF3C7" if pk else ("#E0F2FE" if name in fk_cols else "white")
        rows.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="{bg}"><FONT POINT-SIZE="9">{name}</FONT></TD>'
                    f'<TD ALIGN="LEFT" BGCOLOR="{bg}"><FONT POINT-SIZE="9" COLOR="#555555">{typ}</FONT></TD>'
                    f'<TD BGCOLOR="{bg}"><FONT POINT-SIZE="9"><B>{tag or "&nbsp;"}</B></FONT></TD></TR>')
    return (f'{t} [shape=plain, label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" COLOR="{TEAL}">'
            f'<TR><TD COLSPAN="3" BGCOLOR="{TEAL}"><FONT COLOR="white" POINT-SIZE="11"><B>{t}</B></FONT></TD></TR>'
            + "".join(rows) + "</TABLE>>];")


edges = []
for t, lst in fks.items():
    for col, ref_t, ref_c in lst:
        edges.append(f'{t} -> {ref_t} [label="{col}", fontsize=8, arrowhead=none, arrowtail=crow, dir=both, color="{DARK}"];')
dot("database_diagram", f"""
digraph G {{
  rankdir=LR; nodesep=0.35; ranksep=1.0; {FONT}; bgcolor="white";
  {chr(10).join(table_node(t) for t in tables)}
  {chr(10).join(edges)}
}}
""")

entities = {
    "users": "User", "diseases": "Disease", "symptoms": "Symptom", "medicines": "Medicine",
    "care_guidance": "Care Guidance", "recommendation_queries": "Recommendation Query",
    "recommendation_details": "Recommendation Detail", "feedback": "Feedback", "datasets": "Dataset", "ml_models": "ML Model",
    "disease_symptom_map": "Disease-Symptom Map", "disease_medicine_map": "Disease-Medicine Map",
    "sessions": "Session", "password_resets": "Password Reset", "audit_log": "Audit Log"}
erd_edges = [e for e in edges]
erd_nodes = "\n".join(
    f'{t} [shape=box, style="filled", fillcolor="{"#FEF3C7" if t.endswith("_map") or t=="recommendation_details" else LIGHT}", color="{TEAL}", label="{entities[t]}"];' for t in tables)
dot("erd", f"""
digraph G {{
  rankdir=LR; nodesep=0.4; ranksep=0.9; {FONT}; bgcolor="white";
  node [{FONT}, fontsize=12, margin="0.18,0.1"];
  {erd_nodes}
  {chr(10).join(e.replace(', fontsize=8', ', fontsize=9') for e in erd_edges)}
}}
""")

# ---------------------------------------------------------------- Gantt chart (SRS work plan)
tasks = [("Project initiation & requirements gathering", 1, 2), ("SRS documentation", 2, 3),
         ("Data collection & pre-processing", 3, 5), ("Model selection & architecture design", 5, 6),
         ("Model training & validation", 6, 9), ("Hyper-parameter tuning & optimisation", 9, 10),
         ("Recommendation engine development", 10, 12), ("Web interface development (Streamlit)", 11, 13),
         ("Integration & system testing", 13, 14), ("User acceptance testing (UAT)", 14, 15),
         ("Bug fixing & final optimisation", 15, 16), ("Documentation & final report", 16, 17),
         ("Project presentation & demo", 17, 18)]
fig, ax = plt.subplots(figsize=(11, 5.2))
for i, (name, a, b) in enumerate(tasks):
    ax.barh(i, b - a + 1, left=a - 0.5, color=TEAL if i % 2 == 0 else "#14B8A6", edgecolor="white", height=0.62)
    ax.text(a - 0.5 + (b - a + 1) / 2, i, f"W{a}-{b}" if a != b else f"W{a}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([f"{i + 1}. {t[0]}" for i, t in enumerate(tasks)], fontsize=9)
ax.invert_yaxis()
ax.set_xticks(range(1, 19)); ax.set_xticklabels([f"W{i}" for i in range(1, 19)], fontsize=8)
ax.set_xlim(0.5, 18.5); ax.grid(axis="x", color="#E5E7EB"); ax.set_axisbelow(True)
ax.set_title("Work plan (Gantt chart) - 18 weeks, Spring 2026", fontsize=12, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(HERE / "gantt.png", dpi=170); plt.close()
print("made gantt")

# ---------------------------------------------------------------- sequence diagrams
def wrap(t, n=34):
    return "\n".join(textwrap.wrap(t, n)) or t


def seq(name, title, parts, msgs):
    n = len(parts)
    xs = {p: 1.6 + i * (11.6 / max(n - 1, 1)) * (n - 1) / n * 0.95 for i, p in enumerate(parts)}
    spacing = 11.2 / n
    xs = {p: 0.9 + spacing * (i + 0.5) for i, p in enumerate(parts)}
    H = 1.6 + sum(0.62 + 0.18 * (len(wrap(m[2]).split("\n")) - 1) if m[0] != "sep" else 0.5 for m in msgs)
    fig, ax = plt.subplots(figsize=(12.2, H))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, H); ax.axis("off")
    ax.text(6.1, H - 0.3, title, ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
    top = H - 0.75
    for p in parts:
        ax.add_patch(FancyBboxPatch((xs[p] - 0.85, top - 0.28), 1.7, 0.56, boxstyle="round,pad=0.02", fc=LIGHT, ec=TEAL, lw=1.4))
        ax.text(xs[p], top, p.replace(" / ", "\n").replace("Recommendation Engine", "Recommendation\nEngine").replace("Dataset Manager", "Dataset\nManager").replace("Model Manager", "Model\nManager").replace("Auth Controller", "Auth\nController").replace("Administrator", "Admin-\nistrator").replace("Streamlit UI", "Streamlit\nUI").replace("Active ML Model", "Active\nML Model").replace("Dataset Store", "Dataset\nStore").replace("Model Store", "Model\nStore"), ha="center", va="center", fontsize=8.5, fontweight="bold", color=DARK, linespacing=1.0)
        ax.plot([xs[p], xs[p]], [top - 0.28, 0.2], ls="--", color="#9CA3AF", lw=0.9, zorder=0)
    y = top - 0.65
    for m in msgs:
        if m[0] == "sep":
            ax.plot([0.3, 11.9], [y + 0.1, y + 0.1], ls=":", color=AMBER, lw=1.3)
            ax.text(0.35, y + 0.2, m[1], fontsize=8, color="#B45309", fontweight="bold", va="bottom")
            y -= 0.5
            continue
        a, b, text = m[0], m[1], wrap(m[2])
        ret = len(m) > 3 and m[3] == "ret"
        lines = len(text.split("\n"))
        if a == b:  # self message
            ax.annotate("", xy=(xs[a], y - 0.18), xytext=(xs[a], y + 0.12), arrowprops=dict(arrowstyle="->", color=DARK, connectionstyle="arc3,rad=-1.6"))
            ax.text(xs[a] + 0.32, y - 0.03, text, fontsize=7.8, va="center", ha="left", color="#111827")
        else:
            ax.annotate("", xy=(xs[b], y), xytext=(xs[a], y),
                        arrowprops=dict(arrowstyle="->" if not ret else "->", color=DARK if not ret else "#6B7280", lw=1.3, ls="--" if ret else "-"))
            ax.text((xs[a] + xs[b]) / 2, y + 0.07, text, fontsize=7.8, ha="center", va="bottom", color="#111827" if not ret else "#4B5563",
                    style="italic" if ret else "normal")
        y -= 0.62 + 0.18 * (lines - 1)
    plt.savefig(HERE / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    print("made", name)


seq("seq_uc01_register", "UC-01 User Registration", ["User", "Streamlit UI", "Auth Controller", "Security", "Database"], [
    ("User", "Streamlit UI", "Enter name, email, password, role; tick consent; click Register"),
    ("Streamlit UI", "Auth Controller", "register(name, email, password, confirm, role, consent)"),
    ("Auth Controller", "Auth Controller", "validate email format, password strength, consent"),
    ("Auth Controller", "Database", "SELECT user WHERE email = ?"),
    ("Database", "Auth Controller", "none found", "ret"),
    ("sep", "[alternative: email already registered]"),
    ("Auth Controller", "Streamlit UI", "'Email already registered.' + prompt to log in", "ret"),
    ("sep", "[main flow continues]"),
    ("Auth Controller", "Security", "hash_password(password) - salted PBKDF2-SHA256"),
    ("Security", "Auth Controller", "hash, salt", "ret"),
    ("Auth Controller", "Database", "INSERT user (hash, role, consent_at)"),
    ("Auth Controller", "Database", "log simulated confirmation e-mail"),
    ("Auth Controller", "Streamlit UI", "success", "ret"),
    ("Streamlit UI", "User", "login page with success message", "ret")])

seq("seq_uc02_login", "UC-02 User Login", ["User", "Streamlit UI", "Auth Controller", "Security", "Database"], [
    ("User", "Streamlit UI", "Enter email and password; click Login"),
    ("Streamlit UI", "Auth Controller", "login(email, password)"),
    ("Auth Controller", "Database", "SELECT user WHERE email = ?"),
    ("Database", "Auth Controller", "user row", "ret"),
    ("Auth Controller", "Auth Controller", "check is_active and locked_until"),
    ("Auth Controller", "Security", "verify_password(password, hash, salt)"),
    ("sep", "[alternative: wrong password]"),
    ("Auth Controller", "Database", "failed_attempts + 1 (lock 15 min at the 5th)"),
    ("Auth Controller", "Streamlit UI", "'Invalid email or password.'", "ret"),
    ("sep", "[main flow: correct password]"),
    ("Auth Controller", "Database", "reset counter; INSERT session (token hash, 8 h expiry)"),
    ("Auth Controller", "Streamlit UI", "session token", "ret"),
    ("Streamlit UI", "User", "recommendation dashboard (AI chatbot)", "ret")])

seq("seq_uc03_recommend", "UC-03 Describe Symptoms and Get Recommendations",
    ["User", "Streamlit UI", "Recommendation Engine", "Active ML Model", "Database"], [
    ("User", "Streamlit UI", "Type symptoms in plain English (optionally age, gender, allergy)"),
    ("Streamlit UI", "Recommendation Engine", "recommend(text, previous profile)"),
    ("Recommendation Engine", "Recommendation Engine", "read age / gender / allergy / pregnancy from the sentence"),
    ("Recommendation Engine", "Recommendation Engine", "NLP: synonyms, phrase match, fuzzy match -> symptom flags"),
    ("sep", "[alternative: no symptom recognised -> ask the user to rephrase]"),
    ("Recommendation Engine", "Active ML Model", "predict_proba(symptom vector)"),
    ("Active ML Model", "Recommendation Engine", "probability per disease (top 3)", "ret"),
    ("Recommendation Engine", "Database", "get medicines, relevance priors, precautions, diet, workout"),
    ("Database", "Recommendation Engine", "reference data", "ret"),
    ("Recommendation Engine", "Recommendation Engine", "rank medicines; allergy flags; Rx flag; if confidence < 50 % add warning"),
    ("Recommendation Engine", "Streamlit UI", "Recommendation", "ret"),
    ("Streamlit UI", "Database", "save query + ranked details (AES-256 encrypted)"),
    ("Streamlit UI", "User", "chat reply with feedback prompt", "ret"),
    ("sep", "[exception: model error -> 'Recommendation service is temporarily unavailable.' and error logged]")])

seq("seq_uc04_history", "UC-04 View Recommendation History", ["User", "Streamlit UI", "Database"], [
    ("User", "Streamlit UI", "Open 'My History'"),
    ("Streamlit UI", "Database", "SELECT queries WHERE user_id = ? ORDER BY date"),
    ("Database", "Streamlit UI", "encrypted rows -> decrypted list", "ret"),
    ("sep", "[alternative: no rows]"),
    ("Streamlit UI", "User", "'No recommendation history found. Start by entering your symptoms.'", "ret"),
    ("sep", "[main flow]"),
    ("Streamlit UI", "User", "list: date, disease, confidence, symptoms, top recommendation", "ret"),
    ("User", "Streamlit UI", "Expand one entry"),
    ("Streamlit UI", "Database", "SELECT ranked medicines for query_id"),
    ("Database", "Streamlit UI", "medicine details", "ret"),
    ("Streamlit UI", "User", "full recommendation detail", "ret")])

seq("seq_uc05_feedback", "UC-05 Provide Feedback", ["User", "Streamlit UI", "Database"], [
    ("User", "Streamlit UI", "Open 'Rate this recommendation'"),
    ("User", "Streamlit UI", "Choose 1-5 stars, optional comment, click Submit"),
    ("Streamlit UI", "Streamlit UI", "check rating is between 1 and 5"),
    ("Streamlit UI", "Database", "UPSERT feedback (rating, encrypted comment) for query_id"),
    ("Database", "Streamlit UI", "saved", "ret"),
    ("Streamlit UI", "User", "'Thank you for your feedback!'", "ret"),
    ("sep", "[alternative: form closed without Submit -> nothing is stored]"),
    ("sep", "[exception: save fails -> error message, user may retry, error logged]")])

seq("seq_uc06_dataset", "UC-06 Manage Datasets (Administrator)", ["Administrator", "Streamlit UI", "Dataset Manager", "Dataset Store", "Database"], [
    ("Administrator", "Streamlit UI", "Choose a CSV / JSON file; click Validate and store"),
    ("Streamlit UI", "Dataset Manager", "upload(filename, bytes, admin_id)"),
    ("Dataset Manager", "Dataset Manager", "check size <= 50 MB"),
    ("Dataset Manager", "Dataset Manager", "validate format and required columns (prognosis + symptoms)"),
    ("sep", "[alternative: invalid -> 'Invalid file format. Please upload a CSV or JSON file with required columns.']"),
    ("Dataset Manager", "Dataset Manager", "clean: missing values, binarise, remove duplicates"),
    ("Dataset Manager", "Dataset Store", "save file"),
    ("Dataset Manager", "Database", "INSERT dataset (rows, columns, symptoms, diseases, duplicates)"),
    ("Dataset Manager", "Streamlit UI", "row / column summary", "ret"),
    ("Streamlit UI", "Administrator", "success message with summary", "ret")])

seq("seq_uc07_retrain", "UC-07 Retrain Recommendation Model (Administrator)",
    ["Administrator", "Streamlit UI", "Model Manager", "Trainer / ML Model", "Model Store", "Database"], [
    ("Administrator", "Streamlit UI", "Select dataset and model types; click Train and evaluate"),
    ("Streamlit UI", "Model Manager", "retrain(dataset_id, models, tune)"),
    ("Model Manager", "Model Manager", "lock: no other retraining running"),
    ("Model Manager", "Trainer / ML Model", "clean, split, augment, tune (GridSearchCV), 5-fold CV, evaluate"),
    ("Trainer / ML Model", "Model Manager", "best model + accuracy, precision, recall, F1", "ret"),
    ("Model Manager", "Model Store", "save .pkl bundle"),
    ("Model Manager", "Database", "INSERT model as 'candidate'; read current production metrics"),
    ("Model Manager", "Streamlit UI", "comparison table and recommendation", "ret"),
    ("sep", "[alternative: new accuracy lower -> warning, keep current model recommended]"),
    ("Administrator", "Streamlit UI", "Click Deploy model"),
    ("Streamlit UI", "Model Manager", "deploy(model_id)"),
    ("Model Manager", "Database", "new -> production, previous production -> archived"),
    ("Streamlit UI", "Administrator", "deployment confirmed", "ret"),
    ("sep", "[exception: training crashes -> error logged, previous model stays in production]")])
print("all diagrams done")
