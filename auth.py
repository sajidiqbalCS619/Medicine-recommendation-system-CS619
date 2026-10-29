"""
auth.py - Auth Controller (UC-01 registration, UC-02 login, lockout, password reset).

E-mail is SIMULATED: the project has no mail server, so the message that would
be e-mailed is written to the audit log and returned to the UI for display.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import db
import security

SELF_REGISTER_ROLES = ("patient", "provider", "pharmacist")
INVALID_LOGIN = "Invalid email or password."
EMAIL_EXISTS = "Email already registered. Please log in or recover your password."
GENERIC_ERROR = "Something went wrong on our side. The incident has been logged for the administrator."


@dataclass
class Result:
    ok: bool
    message: str
    user: Optional[sqlite3.Row] = None
    token: Optional[str] = None
    extra: Optional[str] = None


def register(full_name: str, email: str, password: str, confirm: str, role: str, consent: bool) -> Result:
    email = (email or "").strip().lower()
    if not full_name or not full_name.strip():
        return Result(False, "Please enter your full name.")
    if not security.valid_email(email):
        return Result(False, "Please enter a valid email address.")
    problems = security.password_problems(password or "")
    if problems:
        return Result(False, "Password is too weak. It needs " + ", ".join(problems) + ".")
    if password != confirm:
        return Result(False, "Password and confirmation do not match.")
    if role not in SELF_REGISTER_ROLES:
        return Result(False, "Please choose a role: Patient, Healthcare Provider or Pharmacist.")
    if not consent:
        return Result(False, "You must agree to the storage of your data to create an account.")
    try:
        if db.get_user_by_email(email):
            return Result(False, EMAIL_EXISTS)
        uid = db.create_user(full_name, email, password, role, consent=True, verified=True)
    except sqlite3.IntegrityError:
        return Result(False, EMAIL_EXISTS)
    except Exception as exc:  # database failure (UC-01 exception)
        db.log_event("register_failed", repr(exc), level="ERROR")
        return Result(False, GENERIC_ERROR)
    db.log_event("email_simulated", f"To: {email} | Subject: Welcome - your account is confirmed", user_id=uid)
    return Result(True, "Account created successfully. A confirmation email has been sent to "
                        f"{email} (simulated - no mail server is configured). Please log in.")


def login(email: str, password: str) -> Result:
    email = (email or "").strip().lower()
    user = db.get_user_by_email(email)
    if user is None:
        db.log_event("login_failed", f"unknown email {email}", level="WARNING")
        return Result(False, INVALID_LOGIN)
    if not user["is_active"]:
        return Result(False, "This account has been deactivated. Please contact the administrator.")
    if user["locked_until"] and user["locked_until"] > db.now():
        until = user["locked_until"][11:16]
        return Result(False, f"Account temporarily locked after {db.MAX_FAILED_ATTEMPTS} failed attempts "
                             f"(until {until}). Reset your password to unlock it now.")
    if not security.verify_password(password or "", user["password_hash"], user["password_salt"]):
        attempts = user["failed_attempts"] + 1
        if attempts >= db.MAX_FAILED_ATTEMPTS:
            until = (datetime.now() + timedelta(minutes=db.LOCK_MINUTES)).isoformat(timespec="seconds")
            db.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE user_id=?", (attempts, until, user["user_id"]))
            db.log_event("account_locked", email, level="WARNING", user_id=user["user_id"])
            return Result(False, INVALID_LOGIN + f" Your account is now temporarily locked for {db.LOCK_MINUTES} minutes. "
                                                 "You can reset your password to unlock it.")
        db.execute("UPDATE users SET failed_attempts=? WHERE user_id=?", (attempts, user["user_id"]))
        db.log_event("login_failed", email, level="WARNING", user_id=user["user_id"])
        return Result(False, INVALID_LOGIN)
    db.execute("UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE user_id=?", (db.now(), user["user_id"]))
    token = db.create_session(user["user_id"])
    db.log_event("login_ok", email, user_id=user["user_id"])
    return Result(True, "Logged in.", user=db.get_user(user["user_id"]), token=token)


def logout(token: Optional[str]) -> None:
    db.end_session(token)


def request_password_reset(email: str) -> Result:
    """Creates a 30-minute reset code. The code is shown on screen (simulated email)."""
    user = db.get_user_by_email(email)
    msg = "If this email is registered, a reset code has been sent (simulated)."
    if user is None:
        return Result(True, msg)
    code = secrets.token_hex(4).upper()
    exp = (datetime.now() + timedelta(minutes=30)).isoformat(timespec="seconds")
    db.execute("INSERT INTO password_resets(token_hash,user_id,expires_at) VALUES (?,?,?)",
               (security.token_hash(code), user["user_id"], exp))
    db.log_event("email_simulated", f"To: {user['email']} | Subject: Password reset code", user_id=user["user_id"])
    return Result(True, msg, extra=code)


def reset_password(email: str, code: str, new_password: str) -> Result:
    user = db.get_user_by_email(email)
    row = db.q1("SELECT * FROM password_resets WHERE token_hash=? AND used=0 AND expires_at>?",
                (security.token_hash((code or "").strip().upper()), db.now()))
    if user is None or row is None or row["user_id"] != user["user_id"]:
        return Result(False, "Invalid or expired reset code.")
    problems = security.password_problems(new_password or "")
    if problems:
        return Result(False, "Password is too weak. It needs " + ", ".join(problems) + ".")
    h, s = security.hash_password(new_password)
    db.execute("UPDATE users SET password_hash=?, password_salt=?, failed_attempts=0, locked_until=NULL WHERE user_id=?",
               (h, s, user["user_id"]))
    db.execute("UPDATE password_resets SET used=1 WHERE token_hash=?", (row["token_hash"],))
    db.log_event("password_reset", user["email"], user_id=user["user_id"])
    return Result(True, "Password updated. You can log in now.")
