"""
analyze_v2.py
-------------
Zero-GPU diagnostic. Reads the CSVs run_eval() already wrote and tests ONE
hypothesis:

  H: Session-4's authorization training taught "approval-ish token => Low",
     so challenge-v2 adversarial cases that contain approval tokens DECEPTIVELY
     (granted:true, APPROVED, Owner, success) are being UNDER-escalated.

If true, the errors will be directional (predicted severity < true severity)
and concentrated in rows containing approval tokens. That is a different bug
from Session 3 (which OVER-escalated), and it means we traded one keyword
shortcut for another rather than teaching real reasoning.

Run:  python analyze_v2.py
"""

import pandas as pd

SEV = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# tokens our v2 training associated with "authorized -> Low"
APPROVAL_TOKENS = ["granted", "approved", "approval", "authorized", "scheduled",
                   "whitelist", "ticket", "chg-", "aud-", "inc-", "pim",
                   "break_glass", "rules_of_engagement", "policy=", "role="]


def load(name="v2_results_challenge_v2.csv"):
    df = pd.read_csv(name)
    if "ok" not in df.columns:
        df["ok"] = df["risk_level"] == df["pred_risk_level"]
    return df


def direction(df):
    """Is the model under- or over-escalating? Session 3 over-, hypothesis says v2 under-."""
    d = df.dropna(subset=["pred_risk_level"]).copy()
    d = d[d["pred_risk_level"].isin(SEV)]
    d["delta"] = d["pred_risk_level"].map(SEV) - d["risk_level"].map(SEV)
    under = int((d["delta"] < 0).sum())
    over = int((d["delta"] > 0).sum())
    exact = int((d["delta"] == 0).sum())
    print("=" * 62)
    print("ERROR DIRECTION  (pred severity vs true severity)")
    print("=" * 62)
    print(f"  under-escalated (missed danger) : {under}")
    print(f"  over-escalated  (false alarm)   : {over}")
    print(f"  exact                           : {exact}")
    print(f"  mean signed delta               : {d['delta'].mean():+.2f}")
    print("\n  Session 3 failed by OVER-escalating. If this is negative, we flipped")
    print("  the bias instead of fixing it.")
    return d


def approval_token_effect(d):
    """The core test: do rows containing approval tokens get downgraded?"""
    d = d.copy()
    d["has_approval_token"] = d["raw_log"].str.lower().apply(
        lambda s: any(t in s for t in APPROVAL_TOKENS))
    print("\n" + "=" * 62)
    print("HYPOTHESIS TEST: does an approval-ish token drag predictions to Low?")
    print("=" * 62)
    g = d.groupby("has_approval_token").agg(
        n=("ok", "size"), acc_pct=("ok", lambda s: round(s.mean() * 100, 1)),
        mean_delta=("delta", "mean"))
    g["mean_delta"] = g["mean_delta"].round(2)
    print(g.to_string())

    # the decisive cut: TRUE-Critical rows that contain a deceptive approval token
    crit = d[d["risk_level"] == "Critical"]
    ct = crit[crit["has_approval_token"]]
    cn = crit[~crit["has_approval_token"]]
    print(f"\n  TRUE-Critical WITH approval token : {len(ct)} rows, "
          f"{round(ct['ok'].mean() * 100, 1) if len(ct) else 0}% correct")
    print(f"  TRUE-Critical WITHOUT             : {len(cn)} rows, "
          f"{round(cn['ok'].mean() * 100, 1) if len(cn) else 0}% correct")
    print("\n  If the WITH group is much worse, the shortcut is confirmed:")
    print("  the model reads the approval token, not the actor's legitimacy.")
    return d


def show_misses(d, slice_name="adversarial"):
    m = d[(d["novelty_type"] == slice_name) & (~d["ok"])]
    print("\n" + "=" * 62)
    print(f"EVERY MISS IN [{slice_name}]  ({len(m)} rows)")
    print("=" * 62)
    for _, r in m.iterrows():
        tok = [t for t in APPROVAL_TOKENS if t in str(r["raw_log"]).lower()]
        print(f"\n[#{r['challenge_id']}] true={r['risk_level']} -> pred={r['pred_risk_level']}"
              f"  (delta {int(r['delta']):+d})")
        print(f"  log        : {str(r['raw_log'])[:105]}")
        print(f"  approval tokens present: {tok if tok else 'NONE'}")
        print(f"  model reason: {str(r.get('pred_reason', 'n/a'))[:90]}")
        print(f"  should be  : {str(r['rationale'])[:95]}")


def pii_check(d):
    """Did PII recall survive? (was 0.8 on v1 challenge)"""
    tp = int(((d["unauthorized_pii_access"] == 1) & (d["pred_unauthorized_pii_access"] == 1)).sum())
    fn = int(((d["unauthorized_pii_access"] == 1) & (d["pred_unauthorized_pii_access"] == 0)).sum())
    fp = int(((d["unauthorized_pii_access"] == 0) & (d["pred_unauthorized_pii_access"] == 1)).sum())
    print("\n" + "=" * 62)
    print("UNAUTHORIZED-PII FLAG on OOD set")
    print("=" * 62)
    print(f"  caught (TP): {tp} | MISSED (FN): {fn} | false alarms (FP): {fp}")
    if fn:
        print("  -> missed unauthorized PII access is the worst error class for an audit tool.")


if __name__ == "__main__":
    df = load()
    d = direction(df)
    d = approval_token_effect(d)
    show_misses(d, "adversarial")
    pii_check(d)
    print("\n" + "=" * 62)
    print("Read: if errors are UNDER-escalations concentrated in approval-token rows,")
    print("we swapped 'scary word => Critical' for 'approval word => Low'.")
    print("The fix is NOT more of the same data — see Session 5 options.")
