# Medicine Recommendation System - AI Chatbot Assistant

CS619 Final Year Project · Group **S26PROJECTFEF30** · Supervisor: Dr. Mushtaq Hussain · Virtual University of Pakistan

A web-based AI chatbot. The user describes symptoms **in plain English** (optionally mentioning age, gender or allergies in the same
sentence). The system extracts the symptoms with NLP, predicts the most likely condition with a trained machine-learning model, and
replies with ranked medicines (with prescription flag and dosage note), precautions, diet and workout advice. There are no forms or
symptom pickers: the chat box is the only input.

> Academic project. It is **not** a substitute for professional medical advice.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first start creates the SQLite database (`data/mrs.db`), seeds the reference data from `Datasets/`, creates the administrator
account and registers the trained models. Nothing else has to be set up.

| Account | Email | Password |
|---|---|---|
| Administrator (seeded) | `admin@mrs.local` | `Admin@12345` (change via env vars `MRS_ADMIN_EMAIL` / `MRS_ADMIN_PASSWORD` before the first run) |
| Patient / Provider / Pharmacist | create on the **Register** tab | your choice (8+ chars, upper, lower, digit) |

Try typing: *"I am 30, female, allergic to fluconazole, with itching, skin rash and nodal skin eruptions"* or
*"I have high fever, headache, nausea, vomiting, chills and sweating"*.

## What is inside

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: login/register, AI chatbot, history, feedback, model evaluation, admin panel |
| `engine.py` | Recommendation Engine: NLP symptom extraction, profile parsing, prediction, medicine ranking |
| `trainer.py` | Data cleaning, model zoo, GridSearchCV, cross-validation, evaluation, `.pkl` bundles |
| `managers.py` | Dataset Manager (UC-06) and Model Manager (UC-07) |
| `auth.py`, `security.py` | Registration, login, 5-attempt lockout, password reset; PBKDF2 hashing, AES-256-GCM encryption |
| `db.py`, `reference_data.py` | SQLite schema (15 tables), seeding, queries; prescription flags and dosage notes |
| `Notebooks/model_comparison.ipynb` | Final model training/comparison notebook (DT, RF, SVM, XGBoost, MLP, Keras DNN) |
| `Notebooks/training.ipynb` | Prototype notebook (Random Forest) |
| `Models/` | `rf_model.pkl` (prototype), `versions/model_v2_*.pkl` (deployed), comparison tables |
| `Datasets/` | Training data and reference CSV files |
| `tests/` | 34 automated tests, including the 14 test cases of the Design Document |
| `docs/` | Diagrams, screenshots, test results |
| `prototype/app_prototype.py` | The accepted prototype application (unchanged) |

## Running the tests

```bash
pip install pytest
python -m pytest tests -q          # about 2 minutes
python tests/run_tests.py          # writes docs/TEST_RESULTS.md
```

## Re-running the model comparison

Open `Notebooks/model_comparison.ipynb` in Google Colab or Jupyter. It clones this repository when run in Colab. TensorFlow is only
needed for the Keras deep-learning experiment; the app itself does not use it.

## Notes for the examiner

* E-mail confirmation and password-reset e-mails are **simulated** (shown on screen and written to the audit log) because no mail
  server is configured.
* The training data has only 304 unique symptom patterns (4,616 of 4,920 rows are duplicates). Duplicates are removed before
  splitting, so the reported accuracy is not inflated by train/test overlap. See the notebook and report chapter 3.
* The data contains no dosage information. The app shows a generic dosage note per treatment type and never invents doses.
