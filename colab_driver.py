# ==========================================================================
#  COLAB BASELINE DRIVER  —  IT Audit Log Risk Classification (Phase 1)
#  Runtime -> Change runtime type -> T4 GPU, then run the cells below in order.
# ==========================================================================

# ------------------------------------------------------------------ #
# %% CELL 1 — install dependencies
# ------------------------------------------------------------------ #
# !pip install -q -U transformers accelerate torch scikit-learn pandas

# ------------------------------------------------------------------ #
# %% CELL 2 — write the two modules to disk
#   Option A: drag audit_dataset.py + audit_eval_harness.py into the
#             Colab file panel (left sidebar), then skip to CELL 3.
#   Option B: paste each file into its own `%%writefile <name>.py` cell.
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# %% CELL 3 — build the dataset & sanity-check the label distribution
# ------------------------------------------------------------------ #
from audit_dataset import generate_dataset

df = generate_dataset()
print(f"{len(df)} rows")
print(df["risk_level"].value_counts())
df.to_csv("audit_logs.csv", index=False)
df.head()

# ------------------------------------------------------------------ #
# %% CELL 4 — pick a model & load it onto the T4
#   Swap MODEL_ID to compare baselines. All of these fit a 16GB T4 in fp16.
#     "Qwen/Qwen2.5-3B-Instruct"        <- recommended default (best balance)
#     "Qwen/Qwen2.5-1.5B-Instruct"      <- faster / lighter
#     "Qwen/Qwen3-4B-Instruct-2507"     <- strongest zero-shot (a bit slower)
#     "meta-llama/Llama-3.2-1B-Instruct"<- most "tunable" for later fine-tuning
# ------------------------------------------------------------------ #
from audit_eval_harness import load_model, run_baseline, score

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
model, tok = load_model(MODEL_ID)
print("loaded", MODEL_ID)

# ------------------------------------------------------------------ #
# %% CELL 5 — run the baseline (generate + parse + time every row)
# ------------------------------------------------------------------ #
results = run_baseline(df, model, tok)          # real generation on the GPU
metrics = score(results)                         # prints full report

results.to_csv("baseline_results.csv", index=False)

import json
with open("baseline_metrics.json", "w") as f:
    json.dump({"model": MODEL_ID, **metrics}, f, indent=2)
print("\nSaved -> baseline_results.csv  +  baseline_metrics.json")

# ------------------------------------------------------------------ #
# %% CELL 6 (optional) — inspect the rows the model got wrong
# ------------------------------------------------------------------ #
wrong = results[results["risk_level"] != results["pred_risk_level"]]
print(f"{len(wrong)} misclassified rows:")
wrong[["log_id", "raw_log", "risk_level", "pred_risk_level", "parsed_ok"]]
