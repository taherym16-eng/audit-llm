"""
audit_policy.py
---------------
Deterministic severity policy over ENRICHED FACTS (see audit_enrich.py).

DESIGN CLAIM
------------
Once legitimacy is a looked-up fact rather than an inference, the severity
decision is mostly POLICY, not judgement -- and policy belongs in code, where it
is auditable, testable, versionable, and explainable to a regulator. An LLM that
"feels" a log is Critical cannot be put in front of an audit committee. A rule
that says
    is_entitled=False AND resource_class=PII_* AND outcome=SUCCESS -> Critical
can.

This is the honest division of labour the Session-4 data pointed to:
    LLM  -> parse heterogeneous raw formats into fields   (measured strength:
            +33.3 pts over trivial on unseen formats)
    CODE -> entitlement / classification / approval lookups (deterministic)
    CODE -> severity policy                                 (auditable)
    HUMAN-> anything the pipeline marks UNKNOWN             (fail loud)

HONEST CAVEAT
-------------
These rules and the challenge-set labels were written by the same author, so a
high score here partly measures self-consistency, not ground truth. The rules are
written as GENERAL policy (entitlement x sensitivity x outcome x destination),
never tuned per-row. Real validation requires labels from an independent auditor
or production logs. Treat the number as a design demonstration, not a benchmark.
"""

from audit_enrich import enrich_log

PII_CLASSES = {"PII_HR", "PII_CUST", "PII_FIN", "PII_HEALTH"}
SENSITIVE = PII_CLASSES | {"SECRET"}


def decide(facts: dict) -> dict:
    """Enriched facts -> (risk_level, policy_violation, unauthorized_pii_access, reason).
    Rules are ordered: the first match wins, most severe first."""
    role = facts.get("actor_role")
    rclass = facts.get("resource_class")
    entitled = facts.get("is_entitled")
    action = facts.get("action_class")
    outcome = facts.get("outcome")
    dest = facts.get("destination_class")
    vol = facts.get("volume") or 0
    fails = facts.get("failure_count") or 0
    approved = facts.get("approval_valid")

    def out(risk, pol, pii, reason):
        return {"risk_level": risk, "policy_violation": pol,
                "unauthorized_pii_access": pii, "reason": reason}

    # --- R1: destructive actions by a non-authorized role that SUCCEEDED -------
    if facts.get("action_is_destructive") and outcome == "SUCCESS":
        allowed_roles = {"InfraAdmin", "SecurityAdmin", "BreakGlass", "DBA"}
        if role not in allowed_roles or not approved:
            return out("Critical", 1, 0,
                       f"{action} succeeded by role={role} without valid approval")
        return out("Medium", 0, 0, f"{action} by authorized {role} under approval")

    # --- R2: destructive attempt that was DENIED -> attempted, not achieved ----
    if facts.get("action_is_destructive") and outcome == "DENIED":
        return out("High", 1, 0, f"{action} attempted by role={role} but denied")

    # --- R3: sensitive data touched by a NON-entitled role ---------------------
    if rclass in SENSITIVE and entitled is False:
        if outcome == "DENIED":
            return out("High", 1, 0,
                       f"role={role} attempted {rclass} access, denied")
        pii = 1 if rclass in PII_CLASSES or rclass == "SECRET" else 0
        return out("Critical", 1, pii,
                   f"role={role} not entitled to {rclass}; access succeeded")

    # --- R4: sensitive data by an ENTITLED role -------------------------------
    if rclass in SENSITIVE and entitled is True:
        # exfiltration check still applies to entitled roles
        if dest in {"EXTERNAL", "PERSONAL_CLOUD"} and not approved:
            return out("Critical", 1, 1,
                       f"entitled {role} moved {rclass} to {dest} without approval")
        if rclass == "SECRET" and role not in {"SecurityAdmin", "InfraAdmin",
                                               "SvcSecrets", "DBA"}:
            return out("Medium", 0, 0, f"{role} touched {rclass} - monitored")
        return out("Low", 0, 0,
                   f"role={role} is entitled to {rclass}"
                   + (f" (approval {facts['approval_ref']})" if approved else ""))

    # --- R5: data movement to external/personal destinations -------------------
    if action == "DATA_MOVEMENT" and dest in {"EXTERNAL", "PERSONAL_CLOUD"}:
        if approved:
            return out("Low", 0, 0, f"approved transfer to {dest}")
        if vol >= 100_000_000:
            return out("Critical", 1, 1 if rclass in PII_CLASSES else 0,
                       f"{vol/1e9:.1f}GB moved to {dest} without approval")
        return out("High", 1, 0, f"data movement to {dest} without approval")

    # --- R6: brute force ------------------------------------------------------
    if fails >= 5 and outcome == "SUCCESS":
        return out("Critical", 1, 0, f"{fails} failures then success = compromise")
    if fails >= 5 and outcome in {"DENIED", "UNKNOWN"}:
        return out("High", 1, 0, f"{fails} failures, no success = attack in progress")
    if 1 <= fails < 5 and outcome == "SUCCESS":
        return out("Low", 0, 0, "single failure then success = mistyped password")
    if 1 <= fails < 5:
        return out("Medium", 0, 0, "isolated auth failure")

    # --- R7: scans ------------------------------------------------------------
    if action == "SCAN":
        if approved or role in {"SvcScanner", "RedTeam"}:
            return out("Low", 0, 0, f"authorized scan by {role}")
        return out("High", 1, 0, "unauthorized scan / reconnaissance")

    # --- R8: exposure & auth events ------------------------------------------
    if action == "EXPOSURE_CHANGE":
        return out("High", 1, 0, "resource exposed publicly")
    if action == "TOKEN_REPLAY":
        return out("Critical", 1, 0, "token replay = session hijack")
    if action == "MFA_EVENT":
        if fails >= 5 and outcome == "SUCCESS":
            return out("Critical", 1, 0, "MFA fatigue attack succeeded")
        return out("Medium", 1, 0, "MFA control weakened")
    if action == "AUTH_FAILURE":
        return out("Medium", 0, 0, "authentication failure")
    if action == "CONTROL_LAPSE":
        return out("Medium", 1, 0, "security control lapsed")
    if action == "CONFIG_CHANGE":
        return out("Low" if approved else "Medium", 0, 0, "configuration change")
    if action in {"AUTH_SUCCESS", "ROUTINE_OPS", "APP_ERROR", "DATA_READ"}:
        return out("Low", 0, 0, f"{action} by role={role}, nothing sensitive touched")

    return out("Medium", 0, 0, "unclassified - needs review")


def classify(raw: str) -> dict:
    """Full pipeline: raw log -> facts -> decision (+ review flag)."""
    facts = enrich_log(raw)
    d = decide(facts)
    d["needs_human_review"] = facts["needs_human_review"]
    d["facts"] = facts
    return d


if __name__ == "__main__":
    tests = [
        ("CEF:0|PaloAlto|PAN-OS|10.2|end|TRAFFIC|6|src=10.0.7.88 suser=sales_lee "
         "dst=203.0.113.7 dpt=443 app=web-upload bytes_out=5300000000 file=customers_full.csv",
         "Critical"),
        ('{"protoPayload":{"methodName":"logging.sinks.delete","authenticationInfo":'
         '{"principalEmail":"growth_bot@corp.com"}},"severity":"NOTICE"}', "Critical"),
        ('{"protoPayload":{"authenticationInfo":{"principalEmail":"dbadmin@corp.com"},'
         '"methodName":"bigquery.jobs.query","authorizationInfo":[{"granted":true}]},'
         '"resource":{"labels":{"dataset":"hr_ssn"}}}', "Low"),
        ('{"protoPayload":{"authenticationInfo":{"principalEmail":"intern_park@corp.com"},'
         '"methodName":"bigquery.tables.getData","authorizationInfo":[{"granted":true}]},'
         '"resource":{"labels":{"dataset":"payroll_ssn"}}}', "Critical"),
    ]
    for raw, expect in tests:
        d = classify(raw)
        mark = "OK " if d["risk_level"] == expect else "MISS"
        print(f"[{mark}] expect={expect:<9} got={d['risk_level']:<9} :: {d['reason']}")
