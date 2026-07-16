# ==========================================================================
#  run_challenge.py  —  Session 3: true generalization test
#  Scores the fine-tuned model on hand-written, never-before-seen logs and
#  breaks the result down by novelty type (where template-matching shows).
# ==========================================================================

# %% CELL 1 — load base fp16 + LoRA adapter (same as eval_finetuned)
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from audit_eval_harness import run_baseline, score
from challenge_set import generate_challenge_set

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "audit-qlora-adapter"

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# %% CELL 2 — run the fine-tuned model on the challenge set
chal = generate_challenge_set()
results = run_baseline(chal, model, tok)
metrics = score(results)                         # overall report (same format as before)
results.to_csv("challenge_results.csv", index=False)

# %% CELL 3 — the payload: accuracy sliced by novelty type
results["risk_correct"] = results["risk_level"] == results["pred_risk_level"]

print("\n" + "=" * 60)
print("GENERALIZATION BREAKDOWN  (risk_level accuracy by slice)")
print("=" * 60)
for col in ["novelty_type", "difficulty", "category"]:
    g = results.groupby(col)["risk_correct"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(1)
    print(f"\nby {col}:")
    print(g.rename(columns={"mean": "acc_%", "count": "n"}).to_string())

print("\n" + "-" * 60)
print("Headline comparison")
print("-" * 60)
print(f"template test set (Session 2) : 100.0%")
print(f"challenge set (this run)      : {metrics['risk_accuracy'] * 100:.1f}%")
print("The gap between these two numbers is the template-matching effect.")

# %% CELL 4 — error analysis: every miss, with the reasoning it should have used
misses = results[~results["risk_correct"]]
print(f"\n{len(misses)} misclassified of {len(results)}:\n")
for _, row in misses.iterrows():
    print(f"[#{row['challenge_id']}] {row['novelty_type']} | true={row['risk_level']} "
          f"pred={row['pred_risk_level']}")
    print(f"    log   : {row['raw_log'][:100]}")
    print(f"    why   : {row['rationale']}\n")

# also surface flag errors (policy_violation / PII) — often where fp_traps fail
pii_fp = results[(results["unauthorized_pii_access"] == 0) &
                 (results["pred_unauthorized_pii_access"] == 1)]
print(f"False PII-access alarms (authorized activity flagged as unauthorized): "
      f"{len(pii_fp)}")
for _, row in pii_fp.iterrows():
    print(f"  [#{row['challenge_id']}] {row['category']}: {row['raw_log'][:80]}")

misses.to_csv("challenge_misses.csv", index=False)
print("\nSaved -> challenge_results.csv + challenge_misses.csv")
