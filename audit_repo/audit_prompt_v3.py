"""
audit_prompt_v3.py
------------------
Session-5 prompt. The LLM's JOB CHANGES.

Sessions 1-4 asked: "read this log and tell me the severity."
  -> measured result: 48% on novel logs; learned a closed list of breach
     signatures; defaulted unrecognised attacks to Low.

Session 5 asks: "read this log and tell me WHAT HAPPENED, in fields."
  -> plays to the model's ONE measured strength: reading formats it has never
     seen (+33.3 pts over trivial on the format_novel slice).

The model never decides severity. It extracts. Code then looks up entitlement /
classification / approval and applies auditable policy. The model cannot
under-escalate a novel attack it does not recognise, because it is not being
asked to recognise attacks -- only to report actor, resource, action and outcome.

Anything it cannot extract, it must return as null -> the pipeline flags the row
for human review instead of silently guessing Low.
"""

SYSTEM_PROMPT_V3 = """You are a log parser for an enterprise audit pipeline. You read ONE raw \
log line in ANY format (syslog, Apache, JSON, CEF, CloudTrail, GCP, Okta, Azure AD, Kubernetes \
audit, Windows event, auditd, Zeek, RFC5424, Sysmon, EDR) and extract structured facts.

You do NOT assess risk. You do NOT decide if anything is bad. You only report what the log says.

Extract these fields:

  actor            : the account/principal that performed the action, bare username only
                     (e.g. "intern_park" from "intern_park@corp.com", "sales_lee" from
                     "suser=sales_lee"). null if none present.
  resource         : the data/object touched - table, dataset, bucket key, file name,
                     or resource type (e.g. "payroll_ssn", "customers_full.csv",
                     "secrets", "/etc/shadow"). null if none.
  action           : a short verb phrase of what was done, in the log's own terms
                     (e.g. "bigquery.tables.getData", "web-upload", "logging.sinks.delete",
                     "failed password", "mass_file_rename").
  outcome          : exactly one of "SUCCESS", "DENIED", "UNKNOWN".
                     - SUCCESS: the action completed (accepted, granted:true, 200,
                       errorCode null/absent in an API log, bytes actually transferred,
                       "then accepted" after failures).
                     - DENIED : blocked, AccessDenied, 403, res=failed, "no success",
                       "still trying", account locked out.
                     - UNKNOWN: the log genuinely does not say. Use this rather than guessing.
  failure_count    : integer count of failed attempts mentioned, else null
                     (e.g. 41 from "failedAttemptsBefore":"41"; 37 from "pushes_sent":37).
  volume           : bytes or row/record count moved or returned, as an integer, else null.
  destination      : where data went - an IP, hostname, or service (e.g. "203.0.113.7",
                     "mega:personal-backup", "partner-sftp", "backup-nas"). null if none.
  approval_ref     : any change/ticket reference cited (e.g. "CHG-4471", "AUD-220",
                     "SEV1-8842", "INC-3320"). null if none.

Rules:
- Report ONLY what is in the text. Never infer intent, never judge severity.
- If a field is absent or ambiguous, return null. null is CORRECT and useful;
  a guess is not.
- Bare usernames: strip domains and prefixes ("system:serviceaccount:dev:build-runner"
  -> "build-runner").

Respond with ONLY this JSON and nothing else:
{"actor": <string|null>, "resource": <string|null>, "action": <string|null>, "outcome": "<SUCCESS|DENIED|UNKNOWN>", "failure_count": <int|null>, "volume": <int|null>, "destination": <string|null>, "approval_ref": <string|null>}"""

USER_TEMPLATE_V3 = "Extract facts from this log line:\n\n{raw_log}"


def build_messages_v3(raw_log: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V3},
        {"role": "user", "content": USER_TEMPLATE_V3.format(raw_log=raw_log)},
    ]


# --------------------------------------------------------------------------- #
# Few-shot examples: useful for ZERO-SHOT extraction with the BASE model.
# Note we can use the base model here -- no fine-tuning needed, because parsing
# is what it is already good at. That is itself a finding: the fine-tune was
# solving the wrong problem.
# --------------------------------------------------------------------------- #
FEWSHOT = [
    ("CEF:0|PaloAlto|PAN-OS|10.2|end|TRAFFIC|6|src=10.0.7.88 suser=sales_lee "
     "dst=203.0.113.7 dpt=443 app=web-upload bytes_out=5300000000 file=customers_full.csv",
     '{"actor": "sales_lee", "resource": "customers_full.csv", "action": "web-upload", '
     '"outcome": "SUCCESS", "failure_count": null, "volume": 5300000000, '
     '"destination": "203.0.113.7", "approval_ref": null}'),
    ('{"protoPayload":{"authenticationInfo":{"principalEmail":"dbadmin@corp.com"},'
     '"methodName":"bigquery.jobs.query","authorizationInfo":[{"granted":true}]},'
     '"resource":{"labels":{"dataset":"hr_ssn"}}}',
     '{"actor": "dbadmin", "resource": "hr_ssn", "action": "bigquery.jobs.query", '
     '"outcome": "SUCCESS", "failure_count": null, "volume": null, '
     '"destination": null, "approval_ref": null}'),
    ("1720 5 2025-04-08T02:14:00Z gw auth - - [x] 22 failed SSH auths for admin from "
     "45.9.148.30 in 40s, still trying, no success",
     '{"actor": "admin", "resource": null, "action": "failed SSH auth", '
     '"outcome": "DENIED", "failure_count": 22, "volume": null, '
     '"destination": null, "approval_ref": null}'),
]


def build_messages_v3_fewshot(raw_log: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_V3}]
    for raw, gold in FEWSHOT:
        msgs.append({"role": "user", "content": USER_TEMPLATE_V3.format(raw_log=raw)})
        msgs.append({"role": "assistant", "content": gold})
    msgs.append({"role": "user", "content": USER_TEMPLATE_V3.format(raw_log=raw_log)})
    return msgs
