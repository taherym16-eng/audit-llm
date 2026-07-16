"""
audit_dataset.py
-----------------
Generates a synthetic, deterministic dataset of ~70 messy enterprise audit
logs for the "IT Audit Log Risk Classification & PII Detection" pipeline.

Each row carries the RAW log (in one of several realistic, messy formats) plus
three ground-truth labels:
    - risk_level            : Low | Medium | High | Critical
    - policy_violation      : 0 | 1   (corporate policy breached)
    - unauthorized_pii_access : 0 | 1 (an account touched sensitive fields it
                                       has no business role to access)

Design goals:
    * Mixed log formats (syslog, Apache combined, app JSON, Windows-event-ish)
      so the model must actually PARSE, not pattern-match on one schema.
    * A realistic class imbalance (lots of benign traffic, few Critical events).
    * Clearly-labelled "attack" rows: brute-force chains, PII/SSN access by the
      wrong department, leaked API keys, bulk data exfiltration.
"""

import json
import random
from datetime import datetime, timedelta

import pandas as pd

SEED = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ts(base, minutes_offset):
    return (base + timedelta(minutes=minutes_offset)).strftime("%Y-%m-%dT%H:%M:%S")


BENIGN_USERS = ["jsmith", "awong", "rgarcia", "svc_backup", "mpatel", "kobrien"]
MKTG_USERS = ["mktg_dwyer", "sales_lee", "intern_cho", "promo_bot"]
ADMIN_USERS = ["root", "dbadmin", "netops", "sec_admin"]
IPS_INTERNAL = ["10.0.4.12", "10.0.7.88", "192.168.1.34", "172.16.5.9"]
IPS_EXTERNAL = ["203.0.113.7", "198.51.100.23", "45.133.1.14", "185.220.101.4"]
SENSITIVE_TABLES = [
    "employee_ssn", "customer_pii", "payment_cards",
    "api_keys_vault", "hr_salary", "patient_records",
]
NORMAL_TABLES = ["product_catalog", "web_sessions", "campaign_stats", "inventory"]


# --------------------------------------------------------------------------- #
# Row builders  ->  each returns (raw_log, risk_level, policy_violation, pii)
# --------------------------------------------------------------------------- #
def benign_login(base, r):
    u = r.choice(BENIGN_USERS)
    ip = r.choice(IPS_INTERNAL)
    raw = (f"{_ts(base, r.randint(0, 400))} auth_server sshd[{r.randint(1000,9999)}]: "
           f"Accepted publickey for {u} from {ip} port {r.randint(30000,60000)} ssh2")
    return raw, "Low", 0, 0


def benign_web(base, r):
    ip = r.choice(IPS_INTERNAL + IPS_EXTERNAL[:1])
    path = r.choice(["/dashboard", "/api/v2/health", "/reports/daily", "/static/logo.png"])
    raw = (f'{ip} - - [{_ts(base, r.randint(0,400))}] "GET {path} HTTP/1.1" 200 '
           f'{r.randint(200,5000)} "-" "Mozilla/5.0"')
    return raw, "Low", 0, 0


def benign_job(base, r):
    raw = json.dumps({
        "ts": _ts(base, r.randint(0, 400)), "svc": "batch_scheduler",
        "event": "job_completed", "job": r.choice(["nightly_etl", "cache_warm", "log_rotate"]),
        "status": "success", "duration_ms": r.randint(500, 9000),
    })
    return raw, "Low", 0, 0


def benign_db_read(base, r):
    u = r.choice(BENIGN_USERS)
    tbl = r.choice(NORMAL_TABLES)
    raw = (f"{_ts(base, r.randint(0,400))} db_server postgres[{r.randint(100,999)}]: "
           f"LOG: user={u} db=appdb statement: SELECT * FROM {tbl} LIMIT 50")
    return raw, "Low", 0, 0


def single_failed_login(base, r):
    u = r.choice(BENIGN_USERS)
    ip = r.choice(IPS_INTERNAL)
    raw = (f"{_ts(base, r.randint(0,400))} auth_server sshd[{r.randint(1000,9999)}]: "
           f"Failed password for {u} from {ip} port {r.randint(30000,60000)} ssh2")
    return raw, "Medium", 0, 0


def after_hours_access(base, r):
    u = r.choice(BENIGN_USERS)
    night = base.replace(hour=r.randint(1, 4), minute=r.randint(0, 59))
    raw = json.dumps({
        "ts": night.strftime("%Y-%m-%dT%H:%M:%S"), "svc": "vpn_gateway",
        "event": "session_start", "user": u, "src_ip": r.choice(IPS_EXTERNAL),
        "note": "outside_business_hours",
    })
    return raw, "Medium", 1, 0


def config_change(base, r):
    u = r.choice(ADMIN_USERS)
    raw = (f"{_ts(base, r.randint(0,400))} firewall netd: user={u} action=RULE_MODIFY "
           f"rule_id={r.randint(100,400)} change='allow tcp/{r.choice([22,3389,8080])}' approved=yes")
    return raw, "Medium", 0, 0


def expired_cert(base, r):
    raw = (f"{_ts(base, r.randint(0,400))} web_app tls: WARN certificate for "
           f"api.internal expired {r.randint(1,9)} days ago, connection downgraded")
    return raw, "Medium", 1, 0


def repeated_failures_high(base, r):
    """A handful of failures -> High (suspicious but not yet a confirmed breach)."""
    u = r.choice(BENIGN_USERS)
    ip = r.choice(IPS_EXTERNAL)
    n = r.randint(4, 7)
    raw = (f"{_ts(base, r.randint(0,400))} auth_server sshd: {n} failed password attempts "
           f"for {u} from {ip} within 60s; account not locked")
    return raw, "High", 1, 0


def port_scan(base, r):
    ip = r.choice(IPS_EXTERNAL)
    raw = json.dumps({
        "ts": _ts(base, r.randint(0, 400)), "svc": "ids", "event": "port_scan_detected",
        "src_ip": ip, "ports_hit": r.randint(50, 900), "action": "logged",
    })
    return raw, "High", 1, 0


def disabled_account_attempt(base, r):
    u = r.choice(["former_emp01", "contractor_x", "test_acct"])
    raw = (f"{_ts(base, r.randint(0,400))} auth_server: login attempt for DISABLED "
           f"account {u} from {r.choice(IPS_EXTERNAL)} - denied")
    return raw, "High", 1, 0


def sudo_by_nonadmin(base, r):
    u = r.choice(BENIGN_USERS)
    raw = (f"{_ts(base, r.randint(0,400))} db_server sudo: {u} : command not in sudoers ; "
           f"TTY=pts/2 ; PWD=/var ; USER=root ; COMMAND=/bin/cat /etc/shadow")
    return raw, "High", 1, 0


def brute_force_critical(base, r):
    """Many failures FOLLOWED BY a success -> confirmed compromise, Critical."""
    u = r.choice(BENIGN_USERS + ["root"])
    ip = r.choice(IPS_EXTERNAL)
    fails = r.randint(30, 250)
    raw = (f"{_ts(base, r.randint(0,400))} auth_server sshd: {fails} FAILED logins for "
           f"{u} from {ip} then ACCEPTED password for {u} from {ip} - possible brute force success")
    return raw, "Critical", 1, 0


def unauthorized_pii(base, r):
    """Marketing/sales/intern account reading a sensitive HR/PII table."""
    u = r.choice(MKTG_USERS)
    tbl = r.choice(SENSITIVE_TABLES)
    rows = r.randint(500, 90000)
    raw = (f"{_ts(base, r.randint(0,400))} db_server postgres[{r.randint(100,999)}]: "
           f"LOG: user={u} db=corpdb statement: SELECT ssn, dob, full_name FROM {tbl} "
           f"-- returned {rows} rows")
    return raw, "Critical", 1, 1


def leaked_api_key(base, r):
    key = "sk_live_" + "".join(r.choice("abcdef0123456789") for _ in range(24))
    u = r.choice(MKTG_USERS + ["intern_cho"])
    raw = json.dumps({
        "ts": _ts(base, r.randint(0, 400)), "svc": "web_app", "level": "ERROR",
        "user": u, "msg": f"Unhandled exception: DB_CONN_STRING=postgres://admin:{key}@corpdb "
                          f"exposed in stack trace",
    })
    return raw, "Critical", 1, 1


def data_exfil(base, r):
    u = r.choice(MKTG_USERS + ["contractor_x"])
    mb = r.randint(400, 9000)
    raw = (f"{_ts(base, r.randint(0,400))} file_server dlp: user={u} action=DOWNLOAD "
           f"file=customer_pii_export.csv size={mb}MB dest={r.choice(IPS_EXTERNAL)} "
           f"classification=RESTRICTED action_taken=allowed")
    return raw, "Critical", 1, 1


def privilege_escalation(base, r):
    u = r.choice(MKTG_USERS)
    raw = json.dumps({
        "ts": _ts(base, r.randint(0, 400)), "svc": "iam", "event": "role_grant",
        "actor": u, "granted_role": "superuser", "approver": "SELF",
        "note": "self-approved elevation",
    })
    return raw, "Critical", 1, 1


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
# (builder, how_many) — tuned for realistic imbalance, ~70 rows total
COMPOSITION = [
    (benign_login, 8), (benign_web, 8), (benign_job, 6), (benign_db_read, 6),
    (single_failed_login, 5), (after_hours_access, 4), (config_change, 3), (expired_cert, 2),
    (repeated_failures_high, 4), (port_scan, 3), (disabled_account_attempt, 3), (sudo_by_nonadmin, 3),
    (brute_force_critical, 3), (unauthorized_pii, 3), (leaked_api_key, 2),
    (data_exfil, 2), (privilege_escalation, 2),
]


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    r = random.Random(seed)
    base = datetime(2025, 3, 3, 9, 0, 0)
    rows = []
    for builder, count in COMPOSITION:
        for _ in range(count):
            raw, risk, pol, pii = builder(base, r)
            rows.append({
                "raw_log": raw,
                "risk_level": risk,
                "policy_violation": pol,
                "unauthorized_pii_access": pii,
            })
    r.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "log_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Total rows: {len(df)}\n")
    print("risk_level distribution:")
    print(df["risk_level"].value_counts(), "\n")
    print("policy_violation:", dict(df["policy_violation"].value_counts()))
    print("unauthorized_pii_access:", dict(df["unauthorized_pii_access"].value_counts()), "\n")
    print(df.head(8).to_string(index=False))
    df.to_csv("audit_logs.csv", index=False)
    print("\nSaved -> audit_logs.csv")
