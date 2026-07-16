# ==========================================================================
#  eval_finetuned.py  —  re-score the fine-tuned model on the FROZEN test set
#  Reuses the exact Session-1 harness so the number is comparable to 68.7%.
# ==========================================================================

# %% CELL 1 — load base in fp16 + attach the trained LoRA adapter
#   (fp16 base for clean, fast inference; no quantized-merge headaches)
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from audit_eval_harness import run_baseline, score

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "audit-qlora-adapter"

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map={"": 0},
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
print("loaded fine-tuned model")

# %% CELL 2 — evaluate on the frozen 67-row test set
test_df = pd.read_csv("test.csv")
results = run_baseline(test_df, model, tok)
metrics = score(results)

results.to_csv("finetuned_results.csv", index=False)

# %% CELL 3 — before/after summary
BASELINE_ACC = 0.687          # your Session-1 zero-shot number
print("\n================  BEFORE  vs  AFTER  ================")
print(f"zero-shot accuracy : {BASELINE_ACC:.3f}")
print(f"fine-tuned accuracy: {metrics['risk_accuracy']:.3f}")
print(f"delta              : {metrics['risk_accuracy'] - BASELINE_ACC:+.3f}")
print(f"macro-F1 (tuned)   : {metrics['risk_macro_f1']:.3f}")
print(f"parse error rate   : {metrics['parse_error_rate']:.1%}")
