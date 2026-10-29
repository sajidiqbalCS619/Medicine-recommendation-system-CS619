"""Test fixtures: every test run uses a throw-away database, key and model folder."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_tmp = Path(tempfile.mkdtemp(prefix="mrs_tests_"))
os.environ["MRS_DB_PATH"] = str(_tmp / "mrs.db")
os.environ["MRS_DATA_DIR"] = str(_tmp / "data")
os.environ["MRS_DATASET_DIR"] = str(_tmp / "data" / "datasets")
os.environ["MRS_MODEL_DIR"] = str(_tmp / "models")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    db.init_db()
    yield


@pytest.fixture()
def new_user():
    """Create a fresh patient and return (email, password, user_id)."""
    import itertools
    n = next(_counter)
    email, pw = f"user{n}@example.com", "Passw0rdX"
    uid = db.create_user(f"Test User {n}", email, pw, "patient")
    return email, pw, uid


import itertools  # noqa: E402
_counter = itertools.count(1)
