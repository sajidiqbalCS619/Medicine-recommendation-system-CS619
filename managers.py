"""
managers.py - Dataset Manager and Model Manager (Design Document application tier).

UC-06  Manage datasets : validate format/schema, clean, de-duplicate, store, report.
UC-07  Retrain model   : train + evaluate, compare with production, deploy / archive.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

import db
import trainer

MAX_UPLOAD_BYTES = trainer.MAX_UPLOAD_MB * 1024 * 1024
SIZE_MSG = ("Upload exceeds the 50MB size limit. Please compress or split the file and try again.")
LOCK_FILE = Path("data/retrain.lock")
LOCK_STALE_SECONDS = 30 * 60


# ---------------------------------------------------------------------------
# Dataset Manager (UC-06)
# ---------------------------------------------------------------------------
class DatasetManager:
    @staticmethod
    def upload(filename: str, content: bytes, admin_id: int) -> dict:
        """Returns {ok, message, dataset_id, report}.  Never raises for bad input."""
        if len(content) > MAX_UPLOAD_BYTES:
            return {"ok": False, "message": SIZE_MSG}
        try:
            df = trainer.load_dataset(io.BytesIO(content), filename=filename)
            _, _, report = trainer.clean_dataset(df)
        except trainer.DatasetError as exc:
            db.log_event("dataset_rejected", f"{filename}: {exc}", level="WARNING", user_id=admin_id)
            return {"ok": False, "message": str(exc)}
        except Exception as exc:
            db.log_event("dataset_error", f"{filename}: {exc!r}", level="ERROR", user_id=admin_id)
            return {"ok": False, "message": trainer.INVALID_FORMAT_MSG}
        db.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
        stored = db.DATASET_DIR / f"{int(time.time())}_{safe}"
        stored.write_bytes(content)
        did = db.add_dataset(filename, stored, admin_id, report)
        db.log_event("dataset_uploaded", f"{filename} -> id {did}", user_id=admin_id)
        return {
            "ok": True, "dataset_id": did, "report": report,
            "message": (f"Dataset stored: {report['rows_in']} rows, {report['cols_in']} columns "
                        f"({report['n_symptoms']} symptoms, {report['n_diseases']} diseases, "
                        f"{report['duplicates_removed']} duplicate rows removed -> {report['rows_out']} rows used)."),
        }

    @staticmethod
    def load_clean(dataset_id: int):
        row = db.get_dataset(dataset_id)
        if row is None:
            raise trainer.DatasetError("Dataset not found.")
        df = trainer.load_dataset(row["stored_path"])
        return trainer.clean_dataset(df)


# ---------------------------------------------------------------------------
# Model Manager (UC-07)
# ---------------------------------------------------------------------------
def retraining_in_progress() -> bool:
    if LOCK_FILE.exists():
        if time.time() - LOCK_FILE.stat().st_mtime < LOCK_STALE_SECONDS:
            return True
        LOCK_FILE.unlink(missing_ok=True)
    return False


def compare_with_production(candidate_row, production_row) -> dict:
    """Comparison shown to the admin before deployment (UC-07 steps 6-7 and alt path).

    Primary criterion is test accuracy (as in the SRS). If accuracies are within
    0.5 points, accuracy with 3 symptoms decides (the chatbot's real use case).
    """
    if production_row is None:
        return {"recommend_deploy": True, "message": "There is no production model yet: deploying is recommended.",
                "rows": []}
    keys = [("Accuracy", "accuracy"), ("Precision", "precision_w"), ("Recall", "recall_w"), ("F1", "f1_w"),
            ("Accuracy with 3 symptoms", "acc_3_symptoms"), ("5-fold CV accuracy", "cv_accuracy")]
    rows = []
    for label, k in keys:
        new, cur = candidate_row[k], production_row[k]
        rows.append({"Metric": label, "Current production": cur, "New candidate": new,
                     "Change": (new - cur) if (new is not None and cur is not None) else None})
    a_new, a_cur = candidate_row["accuracy"] or 0, production_row["accuracy"] or 0
    s_new, s_cur = candidate_row["acc_3_symptoms"] or 0, production_row["acc_3_symptoms"] or 0
    if a_new < a_cur - 0.005:
        rec, msg = False, ("Warning: the new model's accuracy is LOWER than the current production model. "
                           "Keeping the current model is recommended.")
    elif abs(a_new - a_cur) <= 0.005 and s_new < s_cur - 0.005:
        rec, msg = False, ("Warning: accuracy is equal, but the new model is worse with few symptoms. "
                           "Keeping the current model is recommended.")
    elif a_new > a_cur + 0.005 or s_new > s_cur + 0.005:
        rec, msg = True, "The new model performs better than the current production model. Deploying is recommended."
    else:
        rec, msg = False, "The new model performs the same as the current production model. No benefit in deploying."
    return {"recommend_deploy": rec, "message": msg, "rows": rows}


class ModelManager:
    @staticmethod
    def retrain(dataset_id: int, model_names: List[str], tune: bool, admin_id: int,
                progress=None) -> dict:
        """Train the selected models on the dataset; register the best as a candidate.

        On any crash the production model is untouched (UC-07 exception) and the
        error is logged.
        """
        if retraining_in_progress():
            return {"ok": False, "message": "A retraining job is already running. Please wait for it to finish."}
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.write_text(str(time.time()))
        try:
            X, y, report = DatasetManager.load_clean(dataset_id)
            run = trainer.train_and_compare(X, y, models=model_names, tune=tune, augment=True, progress=progress)
            best = run["best"]
            version = db.next_version()
            slug = re.sub(r"[^a-z0-9]+", "_", best.lower()).strip("_")
            db.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = str(db.MODEL_DIR / f"model_{version}_{slug}.pkl")
            m = trainer.json_safe_metrics(run, best)
            trainer.save_bundle(path, run["results"][best]["estimator"], run["label_encoder"], run["symptom_columns"],
                                {"name": best, "version": version, "metrics": m, "dataset_id": dataset_id})
            mid = db.add_model(version, best, path, dataset_id, m, m.get("best_params", {}), "candidate",
                               f"Trained from dataset #{dataset_id}; compared models: {', '.join(model_names)}.")
            prod = db.production_model()
            cand = [r for r in db.list_models() if r["model_id"] == mid][0]
            cmp_ = compare_with_production(cand, prod)
            db.log_event("model_trained", f"{version} {best} acc={m['test_accuracy']:.3f}", user_id=admin_id)
            return {"ok": True, "model_id": mid, "version": version, "best": best, "table": trainer.metrics_table(run),
                    "comparison": cmp_, "message": f"Model {version} ({best}) trained and evaluated."}
        except Exception as exc:  # training crash: keep the previous model
            db.log_event("training_failed", repr(exc), level="ERROR", user_id=admin_id)
            return {"ok": False, "message": "Training failed. The previous model is still in production. "
                                            f"The error was logged. ({exc})"}
        finally:
            LOCK_FILE.unlink(missing_ok=True)

    @staticmethod
    def deploy(model_id: int, admin_id: int) -> str:
        db.deploy_model(model_id)
        db.log_event("model_deployed", f"model_id={model_id}", user_id=admin_id)
        return "Model deployed. The previous production model has been archived."

    @staticmethod
    def reject(model_id: int, admin_id: int) -> str:
        db.set_model_status(model_id, "rejected")
        db.log_event("model_rejected", f"model_id={model_id}", user_id=admin_id)
        return "Candidate discarded. The current production model is unchanged."

    @staticmethod
    def rollback(model_id: int, admin_id: int) -> str:
        db.deploy_model(model_id)
        db.log_event("model_rollback", f"model_id={model_id}", user_id=admin_id)
        return "Rolled back: the selected archived model is now in production."
