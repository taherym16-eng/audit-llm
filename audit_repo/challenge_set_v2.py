"""
challenge_set_v2.py
-------------------
A SECOND, fresh generalization probe for AFTER Session-4 remediation.

Why a new one: once we train on authorized-twin / attempted-vs-achieved / novel-PII
families, challenge_set.py (v1) becomes a REGRESSION test (did we fix those bugs?).
It no longer measures generalization, because the model has now seen those families.
This set uses formats AND scenarios that appear in neither the training generator nor
challenge v1, so it still answers the real question: did the model learn to AUDIT, or
did it just memorize a bigger set of patterns?

New formats: GCP audit, Azure AD sign-in, Zeek conn.log, Windows 4740/4720/1102-adjacent,
RFC5424 syslog, Sysmon netconn, Palo Alto CSV.
New scenarios (the real test): break-glass emergency access, JIT/PIM approved elevation,
authorized red-team, OAuth refresh-token theft, MFA-fatigue approval, public-bucket
exposure, backdoor service principal, authorized partner DLP transfer.
Labels follow the SAME taxonomy; `rationale` records the deciding cue.
"""

import pandas as pd

CHALLENGE_V2 = [
    # ---- benign / authorized in brand-new formats (fp-trap generalization) ----
    ('{"protoPayload":{"authenticationInfo":{"principalEmail":"dbadmin@corp.com"},"methodName":"bigquery.jobs.query","authorizationInfo":[{"granted":true}]},"resource":{"labels":{"dataset":"hr_ssn"}}}',
     "Low", 0, 0, "db", "fp_trap", "hard",
     "GCP audit: a DBA principal runs an authorized (granted:true) query; role fits resource."),
    ('{"category":"SignInLogs","userPrincipalName":"awong@corp.com","conditionalAccessStatus":"success","riskState":"none","status":{"errorCode":0}}',
     "Low", 0, 0, "auth", "format_novel", "easy",
     "Azure AD sign-in: success, no risk, conditional access satisfied — routine."),
    ('{"event":"break_glass_access","account":"emergency_admin","approver":"CISO","ticket":"SEV1-8842","expires_in":"1h","reason":"prod outage"}',
     "Medium", 0, 0, "priv", "fp_trap", "hard",
     "Break-glass emergency admin use is high-privilege BUT pre-approved and ticketed; notable/monitored, not a violation."),
    ('{"category":"PIMActivation","userPrincipalName":"netops@corp.com","role":"Contributor","approvalState":"approved","justification":"CHG-9910"}',
     "Low", 0, 0, "iam", "fp_trap", "hard",
     "Azure PIM just-in-time elevation, approved with a change ticket — authorized privileged access."),
    ('redteam engagement RT-2025-04: account pentester_svc ran nmap sweep 6000 ports src=10.9.9.9 authorized=true rules_of_engagement=signed',
     "Low", 0, 0, "scan", "fp_trap", "hard",
     "Authorized red-team scan under signed rules of engagement — not a real attack."),
    ('dlp: user=svc_partner_sync action=SEND file=claims_2025.csv classification=RESTRICTED dest=partner-sftp policy=BAA_approved contract=CT-77',
     "Low", 0, 0, "exfil", "fp_trap", "hard",
     "Sensitive data sent to a partner over an APPROVED contractual channel (BAA) — authorized transfer."),
    ('1585 1 2025-04-08T10:00:00Z corp-app web - - [meta env="prod"] user tnguyen viewed own profile page /me',
     "Low", 0, 0, "web", "format_novel", "med",
     "RFC5424 structured syslog: a user viewing their own profile — benign."),

    # ---- medium: soft anomalies, new formats ----
    ('{"category":"SignInLogs","userPrincipalName":"lchen@corp.com","conditionalAccessStatus":"success","riskState":"atRisk","riskLevelDuringSignIn":"low","status":{"errorCode":0}}',
     "Medium", 0, 0, "auth", "semantic_novel", "med",
     "Successful sign-in flagged low-risk by Azure — worth review, not yet an incident."),
    ("<Event><System><EventID>4740</EventID></System><EventData>A user account was locked out.  Account Name: mpatel  Caller Computer: WKS-22</EventData></Event>",
     "Medium", 0, 0, "auth", "format_novel", "med",
     "Account lockout (4740) — could be forgotten password or early attack; soft anomaly on its own."),
    ('{"event":"oauth_consent","user":"kobrien@corp.com","app":"Unverified Third-Party App","scopes":["Mail.Read","Files.ReadWrite.All"],"result":"granted"}',
     "Medium", 1, 0, "auth", "semantic_novel", "hard",
     "User granted broad scopes to an unverified OAuth app — risky consent, soft policy issue."),

    # ---- high: probable attack / attempts / new formats ----
    ("1720 5 2025-04-08T02:14:00Z gw auth - - [x] 22 failed SSH auths for admin from 45.9.148.30 in 40s, still trying, no success",
     "High", 1, 0, "auth", "format_novel", "med",
     "RFC5424: many failed auths, NO success — brute force in progress, not confirmed."),
    ("1712570400.4  C7x  91.202.4.55  55321  10.0.4.12  9200  tcp  -  0  S0  scan  1  40  0  0  (zeek conn.log: many S0 half-open to one host)",
     "High", 1, 0, "scan", "format_novel", "hard",
     "Zeek conn.log showing S0 half-open sweeps from an external host — reconnaissance."),
    ('{"category":"SignInLogs","userPrincipalName":"jsmith@corp.com","location":"Lagos NG","status":{"errorCode":0}} then 4 min later {"userPrincipalName":"jsmith@corp.com","location":"Toronto CA","status":{"errorCode":0}}',
     "High", 0, 0, "auth", "judgment_call", "hard",
     "Two successful sign-ins on different continents minutes apart — impossible travel, likely token/account compromise (arguably Critical)."),
    ('{"eventName":"PutBucketAcl","userIdentity":{"userName":"dev_reed"},"requestParameters":{"bucketName":"corp-analytics","x-amz-acl":"public-read"},"errorCode":null}',
     "High", 1, 0, "db", "semantic_novel", "hard",
     "A dev made an S3 bucket public — serious exposure risk; High (data sensitivity unconfirmed here)."),
    ("<Event><System><EventID>4720</EventID></System><EventData>A user account was created.  New Account: temp_svc99  Subject: helpdesk_omar  Time: 14:20</EventData></Event>",
     "Medium", 0, 0, "iam", "judgment_call", "med",
     "Account creation by helpdesk during the day — routine provisioning unless context says otherwise; low-signal."),

    # ---- critical: confirmed compromise, disguised in new formats ----
    ('{"event":"oauth_token","user":"jsmith@corp.com","grant":"refresh_token","src_ip":"45.9.148.30","note":"refresh token replayed from new ASN 8 min after issuance","result":"access_token_issued"}',
     "Critical", 1, 0, "auth", "adversarial", "hard",
     "A refresh token replayed from a new network to mint an access token = confirmed token theft / session hijack."),
    ('{"event":"mfa_push","user":"rgarcia@corp.com","pushes_sent":37,"final":"APPROVED","src_ip":"185.220.100.8"}',
     "Critical", 1, 0, "auth", "adversarial", "hard",
     "37 MFA push prompts then an APPROVE from a hostile IP = MFA-fatigue attack that succeeded."),
    ('{"protoPayload":{"authenticationInfo":{"principalEmail":"intern_park@corp.com"},"methodName":"bigquery.tables.getData","authorizationInfo":[{"granted":true}]},"resource":{"labels":{"dataset":"payroll_ssn"}}}',
     "Critical", 1, 1, "db", "adversarial", "hard",
     "GCP: an INTERN principal successfully pulled the payroll SSN dataset — authorized by ACL but wrong role = unauthorized PII access."),
    ('{"eventName":"CreateServicePrincipal","userIdentity":{"userName":"growth_bot"},"requestParameters":{"displayName":"backup-helper","role":"Owner"},"errorCode":null,"time":"03:11"}',
     "Critical", 1, 0, "iam", "adversarial", "hard",
     "A marketing bot account created an Owner-role service principal at 3am — backdoor / privilege escalation, succeeded."),
    ("edr: host=SRV-DB01 event=mass_file_rename ext_added=.locked count=48211 process=svchost_fake.exe -- ransomware pattern",
     "Critical", 1, 0, "endpoint", "adversarial", "med",
     "Tens of thousands of files renamed with a ransom extension = active ransomware encryption."),
    ("CEF:0|PaloAlto|PAN-OS|10.2|end|TRAFFIC|6|src=10.0.7.88 suser=sales_lee dst=203.0.113.7 dpt=443 app=web-upload bytes_out=5300000000 file=customers_full.csv",
     "Critical", 1, 1, "exfil", "adversarial", "hard",
     "A sales account uploaded 5GB of full customer data to an external host — confirmed PII exfiltration."),
    ('{"protoPayload":{"methodName":"logging.sinks.delete","authenticationInfo":{"principalEmail":"growth_bot@corp.com"}},"severity":"NOTICE"}',
     "Critical", 1, 0, "endpoint", "adversarial", "hard",
     "Deleting the logging sink (disabling audit logging) by an unexpected account = anti-forensics during compromise."),

    # ---- robustness / distractors ----
    ("Sysmon EventID 3: process=curl.exe dest=api.githubusercontent.com dpt=443 user=devops note=CI dependency fetch",
     "Low", 0, 0, "endpoint", "fp_trap", "med",
     "Outbound connection to a code host by CI/devops — normal dependency fetch, not exfil."),
    ('{"category":"SignInLogs","userPrincipalName":"svc_reporting@corp.com","appDisplayName":"PowerBI","conditionalAccessStatus":"success","riskState":"none","status":{"errorCode":0}}',
     "Low", 0, 0, "auth", "fp_trap", "med",
     "A reporting service account signing into a BI tool it is meant to use — expected automation."),
    ("1712570999.9  C9y  10.0.2.6  49xxx  10.0.4.10  443  tcp  -  0.4  SF  http  ... TRUNCATED",
     "Low", 0, 0, "web", "format_novel", "med",
     "Truncated Zeek record of a normal internal TLS connection — benign despite the cut-off."),
]

COLUMNS = ["raw_log", "risk_level", "policy_violation", "unauthorized_pii_access",
           "category", "novelty_type", "difficulty", "rationale"]


def generate_challenge_set_v2() -> pd.DataFrame:
    df = pd.DataFrame(CHALLENGE_V2, columns=COLUMNS)
    df.insert(0, "challenge_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_challenge_set_v2()
    print(f"challenge v2 rows: {len(df)}\n")
    print(df["risk_level"].value_counts(), "\n")
    print(df["novelty_type"].value_counts())
    df.to_csv("challenge_set_v2.csv", index=False)
    print("\nSaved -> challenge_set_v2.csv")
