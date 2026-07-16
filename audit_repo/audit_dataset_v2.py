"""
audit_dataset_v2.py
-------------------
Session-4 augmented training generator. Reuses the Session-1 builders (via
audit_dataset) and ADDS the counterexample families the model was missing:

  * authorized-but-sensitive access  (hard negatives for the fp_trap failures)
  * attempted vs achieved pairs       (denied -> High, succeeded -> Critical)
  * single-failure "fat finger" logins (Low, vs brute force = Critical)
  * unauthorized PII in NOVEL formats  (S3 GetObject / GraphQL / OUTFILE)
  * the same events recast across formats (CEF / CloudTrail / Okta / EVTX / auditd)
  * benign-but-alarming distractors    (app stack traces, key rotations)

The point is CONTRAST: for every scary pattern the model over-fit to, it now also
sees a benign/authorized twin, so it must learn the discriminating cue (role,
approval, outcome) instead of the surface token.
"""

import json
import random

import pandas as pd

import audit_dataset as ad   # reuse v1 builders, helpers, pools

# ---- v2 pools ------------------------------------------------------------- #
# canonical sensitive tables (fixed here so we don't depend on mutated pools)
SENS = ["employee_ssn", "hr_salary", "customer_pii", "payment_cards", "bank_accounts",
        "tax_records", "patient_records", "medical_claims", "biometric_data",
        "api_keys_vault", "auth_tokens"]

# which role legitimately needs which resource. DBA/Auditor are broad (operational);
# the rest are scoped. This makes the authorized twins ACTUALLY authorized, so the
# only signal separating them from the Critical cases is role/authorization — not the
# table name. (Avoids teaching "approval cue => safe" regardless of role.)
ROLE_TABLES = {
    "DBA": SENS, "Auditor": SENS,
    "HR": ["employee_ssn", "hr_salary", "customer_pii"],
    "Finance": ["payment_cards", "bank_accounts", "tax_records"],
    "Health": ["patient_records", "medical_claims", "biometric_data"],
}
DB_ROLES = [("dbadmin", "DBA"), ("audit_reviewer", "Auditor"), ("hr_analyst", "HR"),
            ("finance_lee", "Finance"), ("ehr_svc", "Health")]
AUTH_SVC = ["svc_payroll", "svc_reporting", "svc_hr_sync", "svc_finance", "svc_backup"]
UNAUTH_USERS = ad.MKTG_USERS + ["intern_park", "intern_cho", "contractor_x",
                                "growth_bot", "promo_bot"]


def _cue(r):
    n = r.randint(100, 900)
    return r.choice([f"approved_by=CISO ticket=AUD-{n}", f"change=CHG-{n} scheduled=true",
                     f"authorized=true window=approved_maintenance", f"policy=whitelist ticket=INC-{n}"])


# ------------------ authorized hard negatives (fp-trap twins) -------------- #
def authorized_sensitive_read(base, r):
    u, role = r.choice(DB_ROLES)
    tbl = r.choice(ROLE_TABLES[role])          # role-appropriate resource only
    if r.random() < 0.5:
        raw = (f"{ad._ts(base, r.randint(0,400))} db_server postgres[{r.randint(100,999)}]: "
               f"LOG: user={u} role={role} statement: SELECT * FROM {tbl} "
               f"WHERE cycle='2025Q{r.randint(1,4)}' -- {_cue(r)}")
    else:
        raw = json.dumps({"svc": r.choice(AUTH_SVC), "principal": u, "role": role,
                          "action": "db.read", "table": tbl, "rows": r.randint(100, 5000),
                          "scheduled": True, "ticket": f"AUD-{r.randint(100,900)}"})
    return raw, "Low", 0, 0


def authorized_bulk_download(base, r):
    raw = (f"{ad._ts(base, r.randint(0,400))} file_server dlp: user={r.choice(AUTH_SVC)} "
           f"action=DOWNLOAD file={r.choice(['customer_pii_export.csv','hr_backup.tar'])} "
           f"size={r.randint(500,9000)}MB dest={r.choice(ad.IPS_INTERNAL)}(backup-nas) "
           f"classification=RESTRICTED policy=backup_whitelist action_taken=allowed")
    return raw, "Low", 0, 0


def scheduled_key_rotation(base, r):
    raw = json.dumps({"svc": "secrets_mgr", "principal": "svc_rotation",
                      "event": "key_rotated", "key_id": r.choice(["api-gw", "db-signing", "s3-kms"]),
                      "old_disabled": True, "new_enabled": True, "scheduled": True})
    return raw, "Low", 0, 0


def approved_admin_cmd(base, r):
    raw = (f"auditd: type=USER_CMD msg=audit({r.randint(1_700_000_000,1_720_000_000)}.1:{r.randint(100,999)}): "
           f"cmd='cat /etc/shadow' acct='{r.choice(ad.ADMIN_USERS)}' res=success "
           f"approved_change=INC-{r.randint(1000,9999)}")
    return raw, "Medium", 0, 0


def authorized_scan(base, r):
    raw = json.dumps({"svc": "vuln_scanner", "account": "svc_nessus", "event": "scan_started",
                      "targets_scanned": r.randint(50, 900), "ports_touched": r.randint(1000, 9000),
                      "authorized": True, "change_ticket": f"CHG-{r.randint(1000,9999)}"})
    return raw, "Low", 0, 0


def benign_app_error(base, r):
    raw = (f"{ad._ts(base, r.randint(0,400))} web_app ERROR: "
           f"java.lang.{r.choice(['NullPointerException','IllegalStateException'])} at "
           f"com.corp.{r.choice(['cart','api','user'])}.Handler.run(Handler.java:{r.randint(20,199)}) "
           f"-- no secrets in trace")
    return raw, "Low", 0, 0


def fat_finger_login(base, r):
    u = r.choice(ad.BENIGN_USERS)
    ip = r.choice(ad.IPS_INTERNAL)
    raw = (f"{ad._ts(base, r.randint(0,400))} auth_server sshd: 1 failed password then Accepted "
           f"password for {u} from {ip} within {r.randint(3,20)}s (single retry)")
    return raw, "Low", 0, 0


# ------------------ attempted vs achieved --------------------------------- #
def denied_privesc(base, r):
    u = r.choice(UNAUTH_USERS)
    raw = json.dumps({"eventName": "AttachUserPolicy", "userIdentity": {"userName": u},
                      "requestParameters": {"policyArn": "arn:aws:iam::aws:policy/AdministratorAccess",
                                            "userName": u}, "errorCode": "AccessDenied"})
    return raw, "High", 1, 0


def achieved_privesc(base, r):
    u = r.choice(UNAUTH_USERS)
    raw = json.dumps({"eventName": "AttachUserPolicy", "userIdentity": {"userName": u},
                      "requestParameters": {"policyArn": "arn:aws:iam::aws:policy/AdministratorAccess",
                                            "userName": u}, "errorCode": None, "responseElements": None})
    return raw, "Critical", 1, 0


def denied_secret_access(base, r):
    raw = json.dumps({"kind": "Event", "verb": "list",
                      "user": {"username": f"system:serviceaccount:dev:{r.choice(['build','ci','app'])}-runner"},
                      "objectRef": {"resource": "secrets", "namespace": "kube-system"},
                      "responseStatus": {"code": 403, "reason": "Forbidden"}})
    return raw, "High", 1, 0


def unauthorized_pii_varied(base, r):
    u = r.choice(UNAUTH_USERS)
    style = r.random()
    rows = r.randint(500, 90000)
    if style < 0.34:
        raw = json.dumps({"eventName": "GetObject", "userIdentity": {"userName": u},
                          "requestParameters": {"bucketName": "corp-hr",
                                                "key": f"payroll/{r.choice(['ssns','salaries','tax'])}_2025.parquet"},
                          "sourceIPAddress": r.choice(ad.IPS_EXTERNAL)})
    elif style < 0.67:
        raw = (f"web_app graphql: principal={u} query=\"{{ customers(first:{rows}){{ fullName "
               f"socialSecurityNumber dateOfBirth }} }}\" returned_rows={rows}")
    else:
        raw = (f"{ad._ts(base, r.randint(0,400))} db_server mysql: user={u} stmt: SELECT iban, "
               f"card_number FROM {r.choice(['bank_accounts','payment_cards'])} INTO OUTFILE '/tmp/x.csv' "
               f"-- {rows} rows")
    return raw, "Critical", 1, 1


# ------------------ brute force across formats ---------------------------- #
def bruteforce_success_varied(base, r):
    u = r.choice(ad.BENIGN_USERS + ["root"])
    ip = r.choice(ad.IPS_EXTERNAL)
    n = r.randint(20, 250)
    style = r.random()
    if style < 0.5:
        raw = json.dumps({"eventType": "user.session.start",
                          "actor": {"alternateId": f"{u}@corp.com"},
                          "client": {"ipAddress": ip}, "outcome": {"result": "SUCCESS"},
                          "debugContext": {"debugData": {"failedAttemptsBefore": str(n)}}})
    else:
        raw = (f"{ad._ts(base, r.randint(0,400))} auth_server sshd: {n} FAILED logins for {u} "
               f"from {ip} then ACCEPTED password for {u}")
    return raw, "Critical", 1, 0


def bruteforce_inprogress_varied(base, r):
    u = r.choice(ad.BENIGN_USERS + ["root"])
    ip = r.choice(ad.IPS_EXTERNAL)
    n = r.randint(8, 40)
    style = r.random()
    if style < 0.5:
        raw = (f"CEF:0|Cisco|ASA|9.1|338002|Multiple failed authentications|5|src={ip} "
               f"suser={u} cnt={n} outcome=failure window=60s account_locked=false")
    else:
        raw = (f"auditd: type=USER_AUTH msg=audit({r.randint(1_700_000_000,1_720_000_000)}.2:{r.randint(100,999)}): "
               f"acct=\"{u}\" addr={ip} res=failed count={n} note=no_success")
    return raw, "High", 1, 0


# ------------------ format-recast benign (format diversity) --------------- #
def recast_benign_login(base, r):
    u = r.choice(ad.BENIGN_USERS)
    style = r.random()
    if style < 0.34:
        raw = json.dumps({"eventName": "ConsoleLogin", "userIdentity": {"userName": u},
                          "additionalEventData": {"MFAUsed": "Yes"},
                          "responseElements": {"ConsoleLogin": "Success"},
                          "sourceIPAddress": r.choice(ad.IPS_INTERNAL)})
    elif style < 0.67:
        raw = (f"<Event><System><EventID>4624</EventID></System><EventData>Logon Type: 3  "
               f"Account Name: {u}  Source Network Address: {r.choice(ad.IPS_INTERNAL)}  "
               f"Elevated Token: No</EventData></Event>")
    else:
        raw = (f"%ASA-6-113039: Group <VPN-Employees> User <{u}> IP <{r.choice(ad.IPS_INTERNAL)}> "
               f"AnyConnect session established")
    return raw, "Low", 0, 0


# --------------------------------------------------------------------------- #
COMPOSITION_V2 = [
    # authorized hard negatives (the fp-trap twins) — generous counts
    (authorized_sensitive_read, 10), (authorized_bulk_download, 5),
    (scheduled_key_rotation, 3), (approved_admin_cmd, 4), (authorized_scan, 4),
    (benign_app_error, 4), (fat_finger_login, 5),
    # attempted vs achieved
    (denied_privesc, 5), (achieved_privesc, 5), (denied_secret_access, 5),
    # PII + brute force in novel formats
    (unauthorized_pii_varied, 8), (bruteforce_success_varied, 6),
    (bruteforce_inprogress_varied, 6),
    # format diversity for benign
    (recast_benign_login, 8),
]


def generate_v2(seed: int) -> pd.DataFrame:
    """Augmented rows only (the NEW families). Combine with v1 for the full train set."""
    r = random.Random(seed)
    base = ad.datetime(2025, 4, 1, 9, 0, 0)
    rows = []
    for builder, count in COMPOSITION_V2:
        for _ in range(count):
            raw, risk, pol, pii = builder(base, r)
            rows.append({"raw_log": raw, "risk_level": risk,
                         "policy_violation": pol, "unauthorized_pii_access": pii})
    r.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_v2(seed=500)
    print(f"v2 augmented rows (one seed): {len(df)}\n")
    print(df["risk_level"].value_counts(), "\n")
    print("policy_violation:", dict(df["policy_violation"].value_counts()))
    print("unauthorized_pii_access:", dict(df["unauthorized_pii_access"].value_counts()))
    print("\nSample authorized-sensitive (hard negatives):")
    for raw in df[df.raw_log.str.contains("role=|scheduled|authorized|whitelist|payroll", case=False)]["raw_log"].head(4):
        print("  -", raw[:120])
