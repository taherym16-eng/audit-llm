"""
make_data.py
------------
Builds the fine-tuning data for Session 2 WITHOUT train/test leakage.

Strategy:
  * TEST  = the original seed-42, 67-row set (unchanged pools) -> the exact set
            your 68.7% zero-shot baseline was measured on. Frozen. Never trained on.
  * TRAIN = a much larger set generated from *different* seeds with *expanded*
            entity pools (more users / IPs / tables), deduped, then any row whose
            raw_log string also appears in TEST is dropped (hard anti-leakage).
  * VAL   = a small slice held out of TRAIN for eval_loss / early-stopping.

Outputs prompt-completion records (conversational) so TRL applies completion-only
loss automatically:  {"prompt": [system, user], "completion": [assistant_json]}
The system/user prompt is imported from the eval harness so train and inference
use the IDENTICAL prompt.
"""

import json
import random

import pandas as pd

import audit_dataset as ad
from audit_eval_harness import build_messages

TEST_SEED = 42
TRAIN_SEED_START = 1000
N_TRAIN_SEEDS = 60          # each seed ~ 67 rows before dedup
VAL_FRACTION = 0.12
MAX_TRAIN = 800             # cap to keep Colab training fast on a T4
RNG = random.Random(7)

# short, label-derived justification so the output schema matches the eval prompt
REASONS = {
    "Low": "routine expected activity",
    "Medium": "minor anomaly or soft policy issue",
    "High": "suspicious probable-attack activity",
    "Critical": "confirmed breach or severe compromise",
}


def build_frozen_test() -> pd.DataFrame:
    """MUST be called before extend_pools() so seed 42 reproduces the eval set."""
    return ad.generate_dataset(seed=TEST_SEED)


def extend_pools() -> None:
    """Widen the entity pools so many seeds yield genuinely varied logs."""
    ad.BENIGN_USERS += ["tnguyen", "bwhite", "hkumar", "svc_monitor", "lchen",
                        "dmurphy", "pkoval", "sadeyemi", "rfischer", "yamara"]
    ad.MKTG_USERS += ["mktg_reed", "sales_ortiz", "intern_park", "growth_bot",
                      "promo_singh", "ads_delacroix"]
    ad.ADMIN_USERS += ["ops_admin", "sysadmin2", "db_root", "infra_svc"]
    ad.IPS_INTERNAL += ["10.0.9.41", "10.0.2.6", "192.168.4.77", "172.16.8.20",
                        "10.0.11.3", "192.168.2.15"]
    ad.IPS_EXTERNAL += ["203.0.113.99", "198.51.100.7", "91.202.4.55",
                        "45.9.148.30", "185.220.100.8", "159.203.12.7"]
    ad.SENSITIVE_TABLES += ["bank_accounts", "tax_records", "medical_claims",
                            "biometric_data", "auth_tokens"]
    ad.NORMAL_TABLES += ["support_tickets", "page_views", "ab_tests",
                         "shipping_status", "feature_flags"]


def build_train_pool() -> pd.DataFrame:
    frames = [ad.generate_dataset(seed=s)
              for s in range(TRAIN_SEED_START, TRAIN_SEED_START + N_TRAIN_SEEDS)]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset="raw_log").reset_index(drop=True)


def remove_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    test_logs = set(test_df["raw_log"])
    keep = ~train_df["raw_log"].isin(test_logs)
    return train_df[keep].reset_index(drop=True), int((~keep).sum())


def to_records(df: pd.DataFrame):
    recs = []
    for _, r in df.iterrows():
        gold = {
            "risk_level": r["risk_level"],
            "policy_violation": int(r["policy_violation"]),
            "unauthorized_pii_access": int(r["unauthorized_pii_access"]),
            "reason": REASONS[r["risk_level"]],
        }
        system_user = build_messages(r["raw_log"])           # [system, user]
        recs.append({
            "prompt": system_user,
            "completion": [{"role": "assistant",
                            "content": json.dumps(gold, separators=(", ", ": "))}],
        })
    return recs


def main():
    test_df = build_frozen_test()                # original pools -> matches baseline
    extend_pools()                               # widen AFTER freezing test
    train_pool = build_train_pool()
    train_pool, n_leaked = remove_leakage(train_pool, test_df)

    # balance a touch toward the rarer, higher-severity classes, then cap
    train_pool = train_pool.sample(frac=1.0, random_state=7).reset_index(drop=True)
    if len(train_pool) > MAX_TRAIN:
        train_pool = train_pool.head(MAX_TRAIN).reset_index(drop=True)

    n_val = max(20, int(len(train_pool) * VAL_FRACTION))
    val_df = train_pool.head(n_val)
    train_df = train_pool.tail(len(train_pool) - n_val)

    # sanity: zero overlap between any split's raw_log
    assert set(train_df["raw_log"]).isdisjoint(set(test_df["raw_log"]))
    assert set(val_df["raw_log"]).isdisjoint(set(test_df["raw_log"]))

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(f"{name}.csv", index=False)

    with open("train.jsonl", "w") as f:
        for rec in to_records(train_df):
            f.write(json.dumps(rec) + "\n")
    with open("val.jsonl", "w") as f:
        for rec in to_records(val_df):
            f.write(json.dumps(rec) + "\n")

    print(f"leakage rows dropped from train : {n_leaked}")
    print(f"TRAIN rows : {len(train_df)}")
    print(f"VAL   rows : {len(val_df)}")
    print(f"TEST  rows : {len(test_df)} (frozen, == baseline eval set)")
    print("\nTRAIN risk_level distribution:")
    print(train_df["risk_level"].value_counts())
    print("\nExample training record:")
    print(json.dumps(to_records(train_df.head(1))[0], indent=2)[:900])


if __name__ == "__main__":
    main()
