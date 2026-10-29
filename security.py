"""
security.py - password hashing and AES-256 field encryption (NFR-04, NFR-09).

* Passwords are never stored: only a salted PBKDF2-HMAC-SHA256 hash.
* Patient data (symptom text, age group, gender, allergy, feedback comments)
  is encrypted with AES-256-GCM before it is written to the database.
* The 256-bit key is read from the MRS_ENCRYPTION_KEY environment variable
  (base64) or, if absent, generated once into data/secret.key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DATA_DIR = Path(os.environ.get("MRS_DATA_DIR", "data"))
PBKDF2_ROUNDS = 200_000

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return (hash_hex, salt_hex)."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return digest.hex(), salt.hex()


def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    digest, _ = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(digest, hash_hex)


def password_problems(password: str) -> list[str]:
    """UC-01 step 5: password strength rules."""
    problems = []
    if len(password) < 8:
        problems.append("at least 8 characters")
    if not re.search(r"[A-Z]", password):
        problems.append("an uppercase letter")
    if not re.search(r"[a-z]", password):
        problems.append("a lowercase letter")
    if not re.search(r"\d", password):
        problems.append("a digit")
    return problems


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# AES-256-GCM field encryption
# ---------------------------------------------------------------------------
_key_cache: bytes | None = None


def _load_key() -> bytes:
    global _key_cache
    if _key_cache is not None:
        return _key_cache
    env = os.environ.get("MRS_ENCRYPTION_KEY")
    if env:
        key = base64.urlsafe_b64decode(env.encode())
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / "secret.key"
        if path.exists():
            key = base64.urlsafe_b64decode(path.read_bytes())
        else:
            key = AESGCM.generate_key(bit_length=256)
            path.write_bytes(base64.urlsafe_b64encode(key))
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    if len(key) != 32:
        raise ValueError("Encryption key must be 32 bytes (AES-256).")
    _key_cache = key
    return key


def reset_key_cache() -> None:  # used by tests
    global _key_cache
    _key_cache = None


def encrypt_text(plain: str | None) -> str | None:
    """AES-256-GCM; output = base64(nonce || ciphertext+tag)."""
    if plain is None:
        return None
    nonce = secrets.token_bytes(12)
    ct = AESGCM(_load_key()).encrypt(nonce, plain.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + ct).decode()


def decrypt_text(blob: str | None) -> str | None:
    if blob is None:
        return None
    raw = base64.urlsafe_b64decode(blob.encode())
    try:
        return AESGCM(_load_key()).decrypt(raw[:12], raw[12:], None).decode("utf-8")
    except Exception:
        return "[unreadable: encryption key changed]"
