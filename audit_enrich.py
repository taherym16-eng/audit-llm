"""
audit_enrich.py
---------------
Session-5 enrichment layer.

WHY THIS EXISTS
---------------
Sessions 1-4 asked a 3B model to INFER, from one raw log line, whether an actor
was authorized. Challenge-v2 showed why that fails: the model learned a closed
list of breach signatures ("Critical == these N patterns") and defaulted novel
attacks to Low. Attack space is open-ended; you cannot enumerate it in training
data.

But a real audit pipeline does not have to guess. It HAS the facts:
  * who the actor is                 -> identity directory (LDAP / Okta / IAM)
  * what role they hold              -> directory
  * how sensitive the resource is    -> data classification catalog
  * whether the role may touch it    -> entitlement matrix
  * whether an approval ref is valid -> change-management system
  * whether the action succeeded     -> parsed from the log

So we STOP asking the model to recognise legitimacy from surface text and start
LOOKING IT UP. Legitimacy becomes an INPUT FIELD, not a guess.

WHAT THIS DOES *NOT* SOLVE (honest scope)
-----------------------------------------
Enrichment shrinks the open-world problem; it does not eliminate it. Entitlement
and outcome become closed-world lookups (reliable). ACTION SEMANTICS still needs
knowledge: a method name we have never seen ("logging.sinks.delete") must be
mapped to an action class. We handle that by returning UNKNOWN explicitly and
escalating for human review -- fail LOUD, not silently-Low, which is exactly the
failure mode we measured in Session 4.
"""

import json
import re

# --------------------------------------------------------------------------- #
# 1. IDENTITY DIRECTORY  (in production: LDAP / Okta / Entra / IAM API)
#    A real company knows every account. Unknown actor -> flagged, not assumed.
# --------------------------------------------------------------------------- #
DIRECTORY = {
    # engineering / ops
    "jsmith": "Engineering", "awong": "Engineering", "rgarcia": "Engineering",
    "mpatel": "Engineering", "kobrien": "Engineering", "tnguyen": "Engineering",
    "lchen": "Engineering", "dmurphy": "Engineering", "pkoval": "Engineering",
    "hkumar": "Engineering", "bwhite": "Engineering", "dev_reed": "Engineering",
    "devops": "Engineering", "netops": "InfraAdmin", "sysadmin2": "InfraAdmin",
    "ops_admin": "InfraAdmin", "infra_svc": "InfraAdmin", "root": "InfraAdmin",
    # data / security / audit
    "dbadmin": "DBA", "db_root": "DBA", "audit_reviewer": "Auditor",
    "sec_admin": "SecurityAdmin", "pentester_svc": "RedTeam",
    "emergency_admin": "BreakGlass",
    # business roles
    "hr_analyst": "HR", "finance_lee": "Finance", "ehr_svc": "Health",
    "helpdesk_omar": "Helpdesk",
    # marketing / sales / interns / contractors  (NOT entitled to PII)
    "mktg_dwyer": "Marketing", "mktg_reed": "Marketing", "promo_singh": "Marketing",
    "ads_delacroix": "Marketing", "growth_bot": "Marketing", "promo_bot": "Marketing",
    "sales_lee": "Sales", "sales_ortiz": "Sales", "intern_cho": "Intern",
    "intern_park": "Intern", "contractor_x": "Contractor",
    # service accounts (scoped to their job)
    "svc_backup": "SvcBackup", "svc_payroll": "SvcPayroll", "svc_hr_sync": "SvcHR",
    "svc_finance": "SvcFinance", "svc_reporting": "SvcReporting",
    "svc_nessus": "SvcScanner", "svc_rotation": "SvcSecrets",
    "svc_partner_sync": "SvcPartner", "svc_monitor": "SvcMonitor",
    "svc_webshop": "SvcApp", "catalog-api": "SvcApp", "build-runner": "SvcCI",
    "ci-runner": "SvcCI", "app-runner": "SvcCI",
    # known-bad / disabled
    "former_emp01": "Disabled", "test_acct": "Disabled", "temp_svc99": "Provisioned",
}

# --------------------------------------------------------------------------- #
# 2. DATA CLASSIFICATION CATALOG  (in production: data catalog / DLP labels)
# --------------------------------------------------------------------------- #
RESOURCE_CLASS = {
    # PII / regulated
    "employee_ssn": "PII_HR", "hr_salary": "PII_HR", "payroll_ssn": "PII_HR",
    "hr_ssn": "PII_HR", "ssns_2025": "PII_HR", "salaries_2025": "PII_HR",
    "customer_pii": "PII_CUST", "customers_full": "PII_CUST",
    "customer_pii_export": "PII_CUST", "web_customers": "PII_CUST",
    "payment_cards": "PII_FIN", "bank_accounts": "PII_FIN", "tax_records": "PII_FIN",
    "tax_2025": "PII_FIN",
    "patient_records": "PII_HEALTH", "medical_claims": "PII_HEALTH",
    "claims_2025": "PII_HEALTH", "biometric_data": "PII_HEALTH",
    # secrets
    "api_keys_vault": "SECRET", "auth_tokens": "SECRET", "shadow": "SECRET",
    "secrets": "SECRET",
    # non-sensitive
    "product_catalog": "PUBLIC", "web_sessions": "INTERNAL", "campaign_stats": "INTERNAL",
    "inventory": "INTERNAL", "support_tickets": "INTERNAL", "page_views": "INTERNAL",
    "ab_tests": "INTERNAL", "shipping_status": "INTERNAL", "feature_flags": "INTERNAL",
    "configmaps": "INTERNAL", "catalog-cfg": "INTERNAL",
}

# --------------------------------------------------------------------------- #
# 3. ENTITLEMENT MATRIX  (in production: IAM entitlement / RBAC export)
#    role -> data classes it may legitimately touch
# --------------------------------------------------------------------------- #
ENTITLEMENTS = {
    "DBA": {"PII_HR", "PII_CUST", "PII_FIN", "PII_HEALTH", "SECRET", "INTERNAL", "PUBLIC"},
    "Auditor": {"PII_HR", "PII_CUST", "PII_FIN", "PII_HEALTH", "INTERNAL", "PUBLIC"},
    "SecurityAdmin": {"SECRET", "INTERNAL", "PUBLIC"},
    "InfraAdmin": {"SECRET", "INTERNAL", "PUBLIC"},
    "BreakGlass": {"SECRET", "INTERNAL", "PUBLIC"},
    "HR": {"PII_HR", "INTERNAL", "PUBLIC"},
    "Finance": {"PII_FIN", "INTERNAL", "PUBLIC"},
    "Health": {"PII_HEALTH", "INTERNAL", "PUBLIC"},
    "SvcPayroll": {"PII_HR", "INTERNAL"},
    "SvcHR": {"PII_HR", "INTERNAL"},
    "SvcFinance": {"PII_FIN", "INTERNAL"},
    "SvcReporting": {"INTERNAL", "PUBLIC"},
    "SvcBackup": {"PII_HR", "PII_CUST", "PII_FIN", "PII_HEALTH", "INTERNAL", "PUBLIC"},
    "SvcPartner": {"PII_HEALTH", "INTERNAL"},
    "SvcSecrets": {"SECRET", "INTERNAL"},
    "SvcScanner": {"INTERNAL", "PUBLIC"},
    "SvcMonitor": {"INTERNAL", "PUBLIC"},
    "SvcApp": {"INTERNAL", "PUBLIC"},
    "SvcCI": {"INTERNAL", "PUBLIC"},
    "RedTeam": {"INTERNAL", "PUBLIC"},
    "Engineering": {"INTERNAL", "PUBLIC"},
    "Helpdesk": {"INTERNAL", "PUBLIC"},
    "Marketing": {"INTERNAL", "PUBLIC"},
    "Sales": {"INTERNAL", "PUBLIC"},
    "Intern": {"PUBLIC"},
    "Contractor": {"PUBLIC"},
    "Disabled": set(),
    "Provisioned": {"PUBLIC"},
}

# --------------------------------------------------------------------------- #
# 4. ACTION SEMANTICS  (the residual open-world surface -> UNKNOWN when unsure)
# --------------------------------------------------------------------------- #
ACTION_CLASS = [
    (r"logging\.sinks\.delete|audit log was cleared|eventid>1102|logs?\s*cleared",
     "AUDIT_LOG_TAMPER"),
    (r"mass_file_rename|\.locked|ransom", "RANSOMWARE"),
    (r"mimikatz|credential_dumping|t1003|lsass", "CREDENTIAL_DUMP"),
    (r"attachuserpolicy|administratoraccess|createserviceprincipal|role_grant|"
     r"granted_role|domain admins|eventid>4732|\"role\":\s*\"owner\"", "PRIV_ESCALATION"),
    (r"refresh_token|token replay|replayed|session hijack", "TOKEN_REPLAY"),
    (r"mfa_push|pushes_sent|mfa.factor.deactivate", "MFA_EVENT"),
    (r"rclone|into outfile|web-upload|action=send|action=download|getobject|"
     r"bytes_out=\d{9,}|dns-tunnel", "DATA_MOVEMENT"),
    (r"select .*from|bigquery|tables\.getdata|jobs\.query|graphql|db\.read", "DATA_READ"),
    (r"port_scan|port sweep|nmap|ports_touched|ports_hit|dest_ports_hit|scan", "SCAN"),
    (r"failed password|failed logins?|failed auth|4625|res=failed|errorcode\":\s*\"?[1-9]",
     "AUTH_FAILURE"),
    (r"accepted publickey|accepted password|consolelogin|session.start|4624|"
     r"anyconnect|signinlogs", "AUTH_SUCCESS"),
    (r"put ?bucket ?acl|public-read", "EXPOSURE_CHANGE"),
    (r"key_rotated|rule_modify|pimactivation|4720", "CONFIG_CHANGE"),
    (r"certificate .*expired|tls", "CONTROL_LAPSE"),
    (r"exception|nullpointer|illegalstate|stack trace", "APP_ERROR"),
    (r"job_completed|health|kube-probe|liveness", "ROUTINE_OPS"),
]

DESTRUCTIVE = {"AUDIT_LOG_TAMPER", "RANSOMWARE", "CREDENTIAL_DUMP",
               "PRIV_ESCALATION", "TOKEN_REPLAY"}

# --------------------------------------------------------------------------- #
# 5. CHANGE MANAGEMENT  (in production: ServiceNow / Jira change API)
# --------------------------------------------------------------------------- #
APPROVAL_RE = re.compile(r"\b(CHG|AUD|INC|SEV1|CT|RT)-?\d{2,}", re.I)


def validate_approval(raw: str):
    """Does the log cite an approval ref, and does it validate? (mocked lookup)"""
    m = APPROVAL_RE.search(raw)
    if not m:
        return {"approval_ref": None, "approval_valid": False}
    # a real system checks the ticket is open, covers this actor, and is in-window
    return {"approval_ref": m.group(0).upper(), "approval_valid": True}


# --------------------------------------------------------------------------- #
# 6. PARSERS  (the LLM can do this too -- see hybrid mode in audit_prompt_v3)
# --------------------------------------------------------------------------- #
ACTOR_PATTERNS = [
    r'"principalEmail"\s*:\s*"([^"@]+)', r'"userPrincipalName"\s*:\s*"([^"@]+)',
    r'"alternateId"\s*:\s*"([^"@]+)', r'"userName"\s*:\s*"([^"]+)"',
    r'"user"\s*:\s*"([^"@]+)', r'"principal"\s*:\s*"([^"]+)"',
    r'"actor"\s*:\s*"([^"@]+)', r'"account"\s*:\s*"([^"]+)"',
    r'serviceaccount:[^:]+:([\w-]+)', r'\buser=([\w\-.]+)', r'\bsuser=([\w\-.]+)',
    r'\bacct=["\']?([\w\-.]+)', r'\bacount=([\w\-.]+)', r'\baccount=([\w\-.]+)',
    r'Account Name:\s*([\w\-.]+)', r'User <([\w\-.]+)>', r'\bfor ([\w\-.]+) from',
    r'Subject:\s*([\w\-.]+)', r'\bprincipal=([\w\-.]+)', r'\bactor["\s:=]+([\w\-.]+)',
]

OUTCOME_DENIED = re.compile(
    r"accessdenied|access denied|\"code\"\s*:\s*40[13]|forbidden|res=failed|"
    r"denied|blocked(?!_but_executed)|failure|not in sudoers|account_expired|"
    r"errorcode\"\s*:\s*\"[a-z]", re.I)
OUTCOME_SUCCESS = re.compile(
    r"success|accepted|granted\"?\s*:\s*true|\"code\"\s*:\s*200|res=success|"
    r"errorcode\"?\s*:\s*null|approved|allowed|established|issued|completed|"
    r"200 \d+|action_taken=allowed", re.I)


def parse_actor(raw: str):
    for pat in ACTOR_PATTERNS:
        m = re.search(pat, raw, re.I)
        if m:
            a = m.group(1).strip().strip('"').lower()
            if a and a not in {"none", "null", "-"}:
                return a
    return None


def parse_resource(raw: str):
    low = raw.lower()
    for key in sorted(RESOURCE_CLASS, key=len, reverse=True):
        if key in low:
            return key
    return None


def parse_action(raw: str):
    low = raw.lower()
    for pat, cls in ACTION_CLASS:
        if re.search(pat, low):
            return cls
    return "UNKNOWN"


def parse_outcome(raw: str):
    """denied beats success: '41 failures then SUCCESS' is still a success, but
    'AccessDenied' with a success-ish word is a denial. Order matters."""
    # negations first: "no success", "not successful", "still trying" are DENIALS
    if re.search(r"no success|not success|unsuccessful|still trying|no_success|"
                 r"account not locked|blocked_but_executed", raw, re.I):
        if re.search(r"blocked_but_executed", raw, re.I):
            return "SUCCESS"          # executed despite the block
        return "DENIED"
    if OUTCOME_DENIED.search(raw):
        # brute-force-then-success: failures present AND a real success marker
        if re.search(r"then accepted|then success|failedattemptsbefore|"
                     r"\"result\"\s*:\s*\"success\"|final\"\s*:\s*\"approved",
                     raw, re.I):
            return "SUCCESS"
        return "DENIED"
    if OUTCOME_SUCCESS.search(raw):
        return "SUCCESS"
    # Convention: structured API audit logs (CloudTrail / GCP protoPayload / Okta)
    # record an error field ONLY on failure. No denial marker + an API method name
    # => the call succeeded.
    if re.search(r"methodName|eventName|protoPayload|\"verb\"|\"event\"\s*:", raw, re.I):
        return "SUCCESS"
    # Data actually moved (bytes/rows counted) => the transfer happened.
    if re.search(r"bytes(?:_out)?[=:\s]+[1-9]|\d+\s*MB|returned_rows[=:\s]*[1-9]|"
                 r"count=[1-9]", raw, re.I):
        return "SUCCESS"
    return "UNKNOWN"


def parse_destination(raw: str):
    if re.search(r"backup-nas|partner-sftp|10\.|192\.168\.|172\.16\.", raw):
        internal = True
    else:
        internal = False
    if re.search(r"mega:|personal-backup|dropbox|external\.cloud|drive\.google", raw, re.I):
        return "PERSONAL_CLOUD"
    if re.search(r"203\.0\.113|198\.51\.100|45\.|185\.|91\.202|159\.203|dest=external",
                 raw):
        return "EXTERNAL"
    return "INTERNAL" if internal else None


def parse_volume(raw: str):
    m = re.search(r"(\d[\d,]*)\s*MB", raw, re.I)
    if m:
        return int(m.group(1).replace(",", "")) * 1_000_000
    m = re.search(r"bytes(?:_out)?[=:\s]+(\d+)", raw, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:returned_rows|rows|count|first:)[=:\s]*(\d{3,})", raw, re.I)
    if m:
        return int(m.group(1))
    return None


def parse_failure_count(raw: str):
    for pat in [r"failedattemptsbefore\"?\s*:\s*\"?(\d+)", r"(\d+)\s*failed",
                r"cnt=(\d+)", r"pushes_sent\"?\s*:\s*(\d+)", r"count=(\d+)"]:
        m = re.search(pat, raw, re.I)
        if m:
            return int(m.group(1))
    return None


# --------------------------------------------------------------------------- #
# 7. THE ENRICHMENT ENTRY POINT
# --------------------------------------------------------------------------- #
def enrich_log(raw: str) -> dict:
    """raw log -> structured audit facts. Unknowns are EXPLICIT, never assumed benign."""
    actor = parse_actor(raw)
    role = DIRECTORY.get(actor, "UNKNOWN_ACTOR" if actor else "NO_ACTOR_PARSED")
    resource = parse_resource(raw)
    rclass = RESOURCE_CLASS.get(resource) if resource else None
    action = parse_action(raw)
    outcome = parse_outcome(raw)

    # THE KEY FIELD: entitlement is LOOKED UP, not inferred from text
    if rclass is None:
        entitled = None                       # no resource identified
    elif role in ENTITLEMENTS:
        entitled = rclass in ENTITLEMENTS[role]
    else:
        entitled = False                      # unknown actor -> not entitled

    facts = {
        "actor": actor,
        "actor_role": role,
        "resource": resource,
        "resource_class": rclass,
        "is_entitled": entitled,
        "action_class": action,
        "action_is_destructive": action in DESTRUCTIVE,
        "outcome": outcome,
        "destination_class": parse_destination(raw),
        "volume": parse_volume(raw),
        "failure_count": parse_failure_count(raw),
    }
    facts.update(validate_approval(raw))
    facts["needs_human_review"] = (
        action == "UNKNOWN" or role in {"UNKNOWN_ACTOR", "NO_ACTOR_PARSED"}
        or outcome == "UNKNOWN"
    )
    return facts


def facts_block(facts: dict) -> str:
    """Render facts for a prompt."""
    keep = ["actor", "actor_role", "resource", "resource_class", "is_entitled",
            "action_class", "action_is_destructive", "outcome", "destination_class",
            "volume", "failure_count", "approval_ref", "approval_valid"]
    return "\n".join(f"  {k}: {facts.get(k)}" for k in keep)


if __name__ == "__main__":
    samples = [
        ("#21 PAN-OS exfil (v2 adapter said Low)",
         "CEF:0|PaloAlto|PAN-OS|10.2|end|TRAFFIC|6|src=10.0.7.88 suser=sales_lee "
         "dst=203.0.113.7 dpt=443 app=web-upload bytes_out=5300000000 file=customers_full.csv"),
        ("#22 logging sink delete (v2 adapter said Low)",
         '{"protoPayload":{"methodName":"logging.sinks.delete","authenticationInfo":'
         '{"principalEmail":"growth_bot@corp.com"}},"severity":"NOTICE"}'),
        ("fp_trap: DBA authorized read (must stay Low)",
         '{"protoPayload":{"authenticationInfo":{"principalEmail":"dbadmin@corp.com"},'
         '"methodName":"bigquery.jobs.query","authorizationInfo":[{"granted":true}]},'
         '"resource":{"labels":{"dataset":"hr_ssn"}}}'),
        ("adversarial: intern pulls payroll SSN (must be Critical)",
         '{"protoPayload":{"authenticationInfo":{"principalEmail":"intern_park@corp.com"},'
         '"methodName":"bigquery.tables.getData","authorizationInfo":[{"granted":true}]},'
         '"resource":{"labels":{"dataset":"payroll_ssn"}}}'),
    ]
    for title, raw in samples:
        print("=" * 70)
        print(title)
        print("-" * 70)
        print(facts_block(enrich_log(raw)))
        print()
