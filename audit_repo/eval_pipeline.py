"""
eval_pipeline.py
----------------
Scores the enrichment + policy pipeline (audit_enrich -> audit_policy) on the
same held-out sets the fine-tuned model was scored on, so the comparison is
apples-to-apples.

READ THE CAVEAT IN audit_policy.py BEFORE QUOTING THESE NUMBERS: the rules and
the challenge labels share an author, so this measures design viability and
self-consistency, not independent ground truth.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from audit_policy import classify
from challenge_set import generate_challenge_set
from challenge_set_v2 import generate_challenge_set_v2

RISK = ["Low", "Medium", "High", "Critical"]
SEV = {r: i for i, r in enumerate(RISK)}

# measured scores of the Session-4 fine-tuned model (adapter v2)
FT_MODEL = {"challenge_v1": 89.7, "challenge_v2": 48.0}
FT_SLICE_V2 = {"adversarial": 14.3, "format_novel": 83.3, "fp_trap": 85.7,
               "judgment_call": 0.0, "semantic_novel": 0.0}


def run(df, name):
    preds = [classify(r) for r in df["raw_log"]]
    out = df.copy()
    out["pred_risk_level"] = [p["risk_level"] for p in preds]
    out["pred_policy_violation"] = [p["policy_violation"] for p in preds]
    out["pred_unauthorized_pii_access"] = [p["unauthorized_pii_access"] for p in preds]
    out["pred_reason"] = [p["reason"] for p in preds]
    out["needs_human_review"] = [p["needs_human_review"] for p in preds]
    out["ok"] = out["risk_level"] == out["pred_risk_level"]
    out["delta"] = out["pred_risk_level"].map(SEV) - out["risk_level"].map(SEV)

    acc = accuracy_score(out["risk_level"], out["pred_risk_level"]) * 100
    mf1 = f1_score(out["risk_level"], out["pred_risk_level"],
                   labels=RISK, average="macro", zero_division=0) * 100
    always_low = (out["risk_level"] == "Low").mean() * 100

    print("=" * 68)
    print(f"{name}   n={len(out)}")
    print("=" * 68)
    print(f"  always-Low baseline : {always_low:5.1f}%")
    print(f"  fine-tuned LLM      : {FT_MODEL.get(name, float('nan')):5.1f}%")
    print(f"  enrich+policy       : {acc:5.1f}%   (macro-F1 {mf1:.1f})")
    print(f"  under-escalations   : {int((out['delta']<0).sum())}  "
          f"over-escalations: {int((out['delta']>0).sum())}")
    print(f"  flagged for review  : {int(out['needs_human_review'].sum())}")

    # unauthorized-PII flag: the metric that matters most for an audit tool
    tp = int(((out.unauthorized_pii_access == 1) & (out.pred_unauthorized_pii_access == 1)).sum())
    fn = int(((out.unauthorized_pii_access == 1) & (out.pred_unauthorized_pii_access == 0)).sum())
    fp = int(((out.unauthorized_pii_access == 0) & (out.pred_unauthorized_pii_access == 1)).sum())
    print(f"  PII flag: caught {tp} | missed {fn} | false alarms {fp}")
    return out


def slices(out, name):
    print(f"\n[{name}] by novelty_type  (vs fine-tuned LLM, vs always-Low):")
    rows = []
    for nt, g in out.groupby("novelty_type"):
        rows.append({
            "slice": nt, "n": len(g),
            "always_Low": round((g.risk_level == "Low").mean() * 100, 1),
            "LLM": FT_SLICE_V2.get(nt, float("nan")) if name == "challenge_v2" else None,
            "pipeline": round(g["ok"].mean() * 100, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    o1 = run(generate_challenge_set(), "challenge_v1")
    print()
    o2 = run(generate_challenge_set_v2(), "challenge_v2")
    slices(o2, "challenge_v2")

    print("\n" + "=" * 68)
    print("MISSES on challenge_v2 (the honest OOD probe)")
    print("=" * 68)
    for _, r in o2[~o2["ok"]].iterrows():
        print(f"\n[#{r['challenge_id']}] {r['novelty_type']} | true={r['risk_level']} "
              f"-> pred={r['pred_risk_level']}  (review_flag={r['needs_human_review']})")
        print(f"  log      : {str(r['raw_log'])[:95]}")
        print(f"  pipeline : {r['pred_reason']}")
        print(f"  expected : {str(r['rationale'])[:90]}")

    o1.to_csv("pipeline_results_challenge_v1.csv", index=False)
    o2.to_csv("pipeline_results_challenge_v2.csv", index=False)
    print("\nsaved -> pipeline_results_challenge_v*.csv")
