"""
make_data_v2.py
---------------
Assembles the Session-4 training data.

  TRAIN = v1 families (broad coverage) + v2 families (the new counterexamples),
          generated from fresh seeds with widened pools, deduped, then leakage-
          stripped against ALL THREE held-out sets:
              * frozen 67-row test   (project-long sanity anchor)
              * challenge v1         (now a REGRESSION test for the fixed bugs)
              * challenge v2         (the NEW generalization probe)
  Prompt  = audit_prompt_v2.build_messages_v2 (train and eval MUST match).

Output: train_v2.jsonl / val_v2.jsonl  (prompt-completion; completion-only loss).
"""

import json
import random

import pandas as pd

import audit_dataset as ad
import audit_dataset_v2 as v2
import make_data as md1                      # reuse extend_pools()
from audit_prompt_v2 import build_messages_v2
from challenge_set import generate_challenge_set
from challenge_set_v2 import generate_challenge_set_v2

TEST_SEED = 42
V1_SEEDS = range(2000, 2045)                  # 45 seeds of v1 families
V2_SEEDS = range(500, 514)                    # 14 seeds of v2 families (~78 rows each)
VAL_FRACTION = 0.12
MAX_TRAIN = 1000
RNG = random.Random(11)


def cue_reason(raw: str, risk: str, pii: int) -> str:
    low = raw.lower()
    if pii:
        return "unauthorized role accessing sensitive personal data"
    if "role=" in low or "approved" in low or "scheduled" in low or "whitelist" in low \
            or "authorized=true" in low or "pim" in low or "break_glass" in low:
        return "actor role/approval authorizes this access"
    if "accessdenied" in low or "403" in low or "res=failed" in low or "denied" in low:
        return "action was denied/blocked — an attempt, not a success"
    if "failedattemptsbefore" in low or "then accepted" in low or "then success" in low \
            or ("failed" in low and "accepted" in low):
        return "repeated failures followed by a success — confirmed compromise"
    return {"Low": "routine expected activity",
            "Medium": "minor anomaly or soft policy issue",
            "High": "suspicious probable-attack activity, unconfirmed",
            "Critical": "confirmed severe compromise"}[risk]


def to_records(df: pd.DataFrame):
    recs = []
    for _, r in df.iterrows():
        gold = {"risk_level": r["risk_level"],
                "policy_violation": int(r["policy_violation"]),
                "unauthorized_pii_access": int(r["unauthorized_pii_access"]),
                "reason": cue_reason(r["raw_log"], r["risk_level"], int(r["unauthorized_pii_access"]))}
        recs.append({"prompt": build_messages_v2(r["raw_log"]),
                     "completion": [{"role": "assistant",
                                     "content": json.dumps(gold, separators=(", ", ": "))}]})
    return recs


def main():
    test_df = ad.generate_dataset(seed=TEST_SEED)     # frozen, original pools
    md1.extend_pools()                                # widen v1 pools for training

    v1_frames = [ad.generate_dataset(seed=s) for s in V1_SEEDS]
    v2_frames = [v2.generate_v2(seed=s) for s in V2_SEEDS]
    pool = pd.concat(v1_frames + v2_frames, ignore_index=True)
    pool = pool.drop_duplicates(subset="raw_log").reset_index(drop=True)

    # hard anti-leakage against every held-out set
    held_out = set(test_df["raw_log"]) \
        | set(generate_challenge_set()["raw_log"]) \
        | set(generate_challenge_set_v2()["raw_log"])
    before = len(pool)
    pool = pool[~pool["raw_log"].isin(held_out)].reset_index(drop=True)
    n_leaked = before - len(pool)

    pool = pool.sample(frac=1.0, random_state=11).reset_index(drop=True)
    if len(pool) > MAX_TRAIN:
        pool = pool.head(MAX_TRAIN).reset_index(drop=True)

    n_val = max(24, int(len(pool) * VAL_FRACTION))
    val_df, train_df = pool.head(n_val), pool.tail(len(pool) - n_val)

    assert set(train_df["raw_log"]).isdisjoint(held_out)
    assert set(val_df["raw_log"]).isdisjoint(held_out)

    with open("train_v2.jsonl", "w") as f:
        for rec in to_records(train_df):
            f.write(json.dumps(rec) + "\n")
    with open("val_v2.jsonl", "w") as f:
        for rec in to_records(val_df):
            f.write(json.dumps(rec) + "\n")
    test_df.to_csv("test.csv", index=False)
    generate_challenge_set().to_csv("challenge_set.csv", index=False)
    generate_challenge_set_v2().to_csv("challenge_set_v2.csv", index=False)

    print(f"leakage rows dropped   : {n_leaked}")
    print(f"TRAIN rows             : {len(train_df)}")
    print(f"VAL   rows             : {len(val_df)}")
    print("\nTRAIN risk_level distribution:")
    print(train_df["risk_level"].value_counts())
    print("\nTRAIN authorized-sensitive (Low) vs unauthorized-PII (Critical) — the key contrast:")
    auth = train_df[train_df["raw_log"].str.contains("role=|scheduled|whitelist|authorized=true|pim",
                                                     case=False) & (train_df["risk_level"] == "Low")]
    print(f"  authorized-sensitive-ish Low rows : {len(auth)}")
    print(f"  unauthorized_pii_access==1 rows   : {int((train_df['unauthorized_pii_access']==1).sum())}")


if __name__ == "__main__":
    main()
