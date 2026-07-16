"""
audit_eval_harness.py
---------------------
Zero-shot baseline evaluation harness for IT audit-log risk classification.

Pipeline:
    load model  ->  build prompt  ->  generate  ->  parse JSON  ->  score

Metrics reported:
    risk_level              : Accuracy, Macro-F1, per-class F1, confusion matrix
    policy_violation        : Precision / Recall / F1  (binary)
    unauthorized_pii_access : Precision / Recall / F1  (binary)
    latency                 : mean, p50, p95 (seconds / row)
    parse_error_rate        : fraction of rows the model returned unparseable JSON

The model-generation step is isolated in `generate_one()` so the parsing and
scoring logic can be unit-tested with a mock generator (no GPU required).
"""

import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

RISK_LABELS = ["Low", "Medium", "High", "Critical"]

# --------------------------------------------------------------------------- #
# 1. PROMPT
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an enterprise IT security audit engine. You read a single raw \
server or application log line and assess it. Logs may be in syslog, Apache access, \
JSON, or Windows-event format, and may be messy or truncated.

Classify the log on three axes:

1. risk_level - exactly one of: Low, Medium, High, Critical
   - Low      : routine, successful, expected activity (normal logins, health checks, benign reads)
   - Medium   : minor anomaly or soft policy issue (a single failed login, after-hours access, expired cert)
   - High     : suspicious / probable-attack activity, not yet confirmed compromise
                (repeated failed logins, port scans, disabled-account attempts, privilege misuse attempts)
   - Critical : confirmed compromise or severe breach (successful brute force, data exfiltration,
                leaked secrets/API keys, unauthorized access to sensitive PII, privilege escalation)

2. policy_violation - 1 if the log shows any corporate/security policy being broken, else 0.

3. unauthorized_pii_access - 1 if an account is accessing sensitive fields it has no
   legitimate role to access (e.g. a marketing/sales/intern account reading SSNs, salaries,
   payment cards, patient records, or an exposed API key / secret), else 0.

Respond with ONLY a single JSON object and nothing else:
{"risk_level": "<Low|Medium|High|Critical>", "policy_violation": <0|1>, "unauthorized_pii_access": <0|1>, "reason": "<short justification>"}"""

USER_TEMPLATE = "Analyze this log line:\n\n{raw_log}"


def build_messages(raw_log: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(raw_log=raw_log)},
    ]


# --------------------------------------------------------------------------- #
# 2. MODEL LOADING  (Colab / T4)
# --------------------------------------------------------------------------- #
def load_model(model_id: str):
    """Load an instruct model + tokenizer onto the GPU in fp16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tok


def generate_one(model, tok, raw_log: str, max_new_tokens: int = 160) -> str:
    """Run one generation. Returns the raw decoded completion (string)."""
    import torch

    messages = build_messages(raw_log)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # greedy -> deterministic baseline
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    # decode only the newly generated tokens
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# 3. OUTPUT PARSING  (robust to fences, preamble, trailing junk)
# --------------------------------------------------------------------------- #
_RISK_ALIASES = {r.lower(): r for r in RISK_LABELS}


def _coerce_flag(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v >= 1 else 0
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 0


def parse_prediction(text: str) -> dict:
    """
    Extract {risk_level, policy_violation, unauthorized_pii_access} from a model
    completion. Falls back to keyword search if JSON is malformed. Returns a dict
    with a `parsed_ok` flag; on total failure risk_level is None.
    """
    result = {"risk_level": None, "policy_violation": 0,
              "unauthorized_pii_access": 0, "parsed_ok": False}

    # find the first {...} block (greedy from first { to last })
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            obj = json.loads(blob)
            rl = str(obj.get("risk_level", "")).strip().lower()
            if rl in _RISK_ALIASES:
                result["risk_level"] = _RISK_ALIASES[rl]
                result["policy_violation"] = _coerce_flag(obj.get("policy_violation", 0))
                result["unauthorized_pii_access"] = _coerce_flag(
                    obj.get("unauthorized_pii_access", 0))
                result["parsed_ok"] = True
                return result
        except json.JSONDecodeError:
            pass

    # fallback: scan raw text for a risk keyword (last mention wins)
    found = None
    for label in RISK_LABELS:
        for hit in re.finditer(rf"\b{label}\b", text, re.IGNORECASE):
            found = label
    if found:
        result["risk_level"] = found
        result["policy_violation"] = 1 if re.search(r"violation|policy", text, re.I) else 0
        result["unauthorized_pii_access"] = 1 if re.search(
            r"unauthor|ssn|pii|api[_ ]?key|exfil", text, re.I) else 0
    return result


# --------------------------------------------------------------------------- #
# 4. EVALUATION LOOP
# --------------------------------------------------------------------------- #
def run_baseline(df: pd.DataFrame, model, tok, generate_fn=generate_one,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Iterate over the dataset, generate + parse + time each row.
    `generate_fn(model, tok, raw_log)` is injectable so it can be mocked.
    Returns the input df augmented with prediction columns.
    """
    preds, latencies = [], []
    for i, row in df.iterrows():
        t0 = time.perf_counter()
        raw_out = generate_fn(model, tok, row["raw_log"])
        latencies.append(time.perf_counter() - t0)
        p = parse_prediction(raw_out)
        preds.append(p)
        if verbose and (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(df)} rows")

    out = df.copy()
    out["pred_risk_level"] = [p["risk_level"] for p in preds]
    out["pred_policy_violation"] = [p["policy_violation"] for p in preds]
    out["pred_unauthorized_pii_access"] = [p["unauthorized_pii_access"] for p in preds]
    out["parsed_ok"] = [p["parsed_ok"] for p in preds]
    out["latency_s"] = latencies
    return out


def _binary_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def score(results: pd.DataFrame) -> dict:
    """Compute all baseline metrics. Unparseable rows count as wrong (fill 'UNK')."""
    y_true = results["risk_level"].tolist()
    y_pred = [p if p in RISK_LABELS else "UNK" for p in results["pred_risk_level"]]

    metrics = {
        "n_rows": len(results),
        "parse_error_rate": round(1 - results["parsed_ok"].mean(), 3),
        "risk_accuracy": round(accuracy_score(y_true, y_pred), 3),
        "risk_macro_f1": round(
            f1_score(y_true, y_pred, labels=RISK_LABELS, average="macro",
                     zero_division=0), 3),
        "policy_violation": _binary_metrics(
            results["policy_violation"], results["pred_policy_violation"]),
        "unauthorized_pii_access": _binary_metrics(
            results["unauthorized_pii_access"], results["pred_unauthorized_pii_access"]),
        "latency_mean_s": round(float(np.mean(results["latency_s"])), 3),
        "latency_p50_s": round(float(np.percentile(results["latency_s"], 50)), 3),
        "latency_p95_s": round(float(np.percentile(results["latency_s"], 95)), 3),
    }

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Rows evaluated       : {metrics['n_rows']}")
    print(f"Parse error rate     : {metrics['parse_error_rate']:.1%}")
    print(f"\nRISK LEVEL")
    print(f"  Accuracy           : {metrics['risk_accuracy']:.3f}")
    print(f"  Macro F1           : {metrics['risk_macro_f1']:.3f}")
    print(f"\nPOLICY VIOLATION (binary)         : {metrics['policy_violation']}")
    print(f"UNAUTHORIZED PII ACCESS (binary)  : {metrics['unauthorized_pii_access']}")
    print(f"\nLATENCY  mean={metrics['latency_mean_s']}s  "
          f"p50={metrics['latency_p50_s']}s  p95={metrics['latency_p95_s']}s")
    print("\nPer-class report (risk_level):")
    print(classification_report(y_true, y_pred, labels=RISK_LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), order",
          RISK_LABELS)
    print(confusion_matrix(y_true, y_pred, labels=RISK_LABELS))
    return metrics
