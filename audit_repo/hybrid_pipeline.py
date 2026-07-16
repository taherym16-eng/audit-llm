"""
hybrid_pipeline.py
------------------
The Session-5 architecture, end to end:

    raw log
       |
       v
  [ LLM extractor ]        <- audit_prompt_v3; plays to the model's measured
       |                      strength (+33.3 pts on unseen formats). Base model
       v                      is fine: no fine-tuning needed for parsing.
  { actor, resource, action, outcome, volume, destination, approval_ref }
       |
       v
  [ deterministic lookups ] <- audit_enrich: identity directory, data catalog,
       |                       entitlement matrix, change-management validation.
       v                       Legitimacy is LOOKED UP, never inferred.
  { actor_role, resource_class, is_entitled, approval_valid, ... }
       |
       v
  [ policy engine ]         <- audit_policy: auditable, versionable rules.
       |
       v
  risk_level + policy_violation + unauthorized_pii_access + reason
  (+ needs_human_review when any field is UNKNOWN -> fail loud, never silent-Low)

Run in Colab (needs GPU for the LLM extractor):
    import hybrid_pipeline
    hybrid_pipeline.run_hybrid_eval()

Or, with no GPU, use the regex extractor to sanity-check the policy layer:
    hybrid_pipeline.run_regex_eval()
"""

import json
import re

import pandas as pd

import audit_enrich as E
from audit_policy import decide
from audit_prompt_v3 import build_messages_v3_fewshot

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #
def load_extractor(model_id: str = MODEL_ID):
    """Base model, fp16 (T4-safe). NO adapter: parsing needs no fine-tuning."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map={"": 0})
    model.eval()
    return model, tok


def extract_with_llm(model, tok, raw_log: str) -> dict:
    import torch
    msgs = build_messages_v3_fewshot(raw_log)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=180, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_extraction(gen)


def parse_extraction(text: str) -> dict:
    """Parse the extractor's JSON. Unparseable -> all-null (=> human review)."""
    blank = {"actor": None, "resource": None, "action": None, "outcome": "UNKNOWN",
             "failure_count": None, "volume": None, "destination": None,
             "approval_ref": None, "extract_ok": False}
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return blank
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return blank
    out = dict(blank)
    for k in ["actor", "resource", "action", "destination", "approval_ref"]:
        v = o.get(k)
        out[k] = str(v).strip().lower() if v not in (None, "", "null") else None
    oc = str(o.get("outcome", "UNKNOWN")).upper()
    out["outcome"] = oc if oc in {"SUCCESS", "DENIED", "UNKNOWN"} else "UNKNOWN"
    for k in ["failure_count", "volume"]:
        v = o.get(k)
        try:
            out[k] = int(v) if v not in (None, "", "null") else None
        except (TypeError, ValueError):
            out[k] = None
    out["extract_ok"] = True
    return out


# --------------------------------------------------------------------------- #
# extracted fields -> enriched facts (the deterministic half)
# --------------------------------------------------------------------------- #
def facts_from_extraction(x: dict) -> dict:
    """Same lookups as audit_enrich.enrich_log, but fed by the LLM's extraction
    instead of regexes. This is the only difference between the two variants."""
    actor = x.get("actor")
    role = E.DIRECTORY.get(actor, "UNKNOWN_ACTOR" if actor else "NO_ACTOR_PARSED")

    # map a free-text resource string onto the data catalog
    rclass, resource = None, x.get("resource")
    if resource:
        for key in sorted(E.RESOURCE_CLASS, key=len, reverse=True):
            if key in resource:
                rclass = E.RESOURCE_CLASS[key]
                break

    action_text = (x.get("action") or "") + " " + (resource or "")
    action = "UNKNOWN"
    for pat, cls in E.ACTION_CLASS:
        if re.search(pat, action_text, re.I):
            action = cls
            break

    if rclass is None:
        entitled = None
    elif role in E.ENTITLEMENTS:
        entitled = rclass in E.ENTITLEMENTS[role]
    else:
        entitled = False

    dest = x.get("destination")
    dest_class = None
    if dest:
        if re.search(r"mega:|personal|dropbox|external\.cloud", dest, re.I):
            dest_class = "PERSONAL_CLOUD"
        elif re.search(r"^(10\.|192\.168\.|172\.16\.)|nas|internal|partner-sftp", dest, re.I):
            dest_class = "INTERNAL"
        else:
            dest_class = "EXTERNAL"

    approval = x.get("approval_ref")
    facts = {
        "actor": actor, "actor_role": role,
        "resource": resource, "resource_class": rclass, "is_entitled": entitled,
        "action_class": action, "action_is_destructive": action in E.DESTRUCTIVE,
        "outcome": x.get("outcome", "UNKNOWN"),
        "destination_class": dest_class,
        "volume": x.get("volume"), "failure_count": x.get("failure_count"),
        "approval_ref": approval, "approval_valid": bool(approval),
    }
    facts["needs_human_review"] = (
        action == "UNKNOWN" or role in {"UNKNOWN_ACTOR", "NO_ACTOR_PARSED"}
        or facts["outcome"] == "UNKNOWN" or not x.get("extract_ok", True)
    )
    return facts


def classify_hybrid(model, tok, raw_log: str) -> dict:
    x = extract_with_llm(model, tok, raw_log)
    facts = facts_from_extraction(x)
    d = decide(facts)
    d["needs_human_review"] = facts["needs_human_review"]
    d["facts"] = facts
    d["extraction"] = x
    return d


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _score(df, preds, name):
    from sklearn.metrics import accuracy_score, f1_score
    RISK = ["Low", "Medium", "High", "Critical"]
    out = df.copy()
    out["pred_risk_level"] = [p["risk_level"] for p in preds]
    out["pred_unauthorized_pii_access"] = [p["unauthorized_pii_access"] for p in preds]
    out["pred_reason"] = [p["reason"] for p in preds]
    out["needs_human_review"] = [p["needs_human_review"] for p in preds]
    out["ok"] = out["risk_level"] == out["pred_risk_level"]
    acc = accuracy_score(out["risk_level"], out["pred_risk_level"]) * 100
    mf1 = f1_score(out["risk_level"], out["pred_risk_level"], labels=RISK,
                   average="macro", zero_division=0) * 100
    print(f"\n{name}: accuracy {acc:.1f}%  macro-F1 {mf1:.1f}  "
          f"flagged-for-review {int(out['needs_human_review'].sum())}/{len(out)}")
    if "novelty_type" in out.columns:
        g = out.groupby("novelty_type")["ok"].agg(["mean", "count"])
        g["mean"] = (g["mean"] * 100).round(1)
        print(g.rename(columns={"mean": "acc_%", "count": "n"}).to_string())
    return out, acc


def run_hybrid_eval():
    """LLM extractor + policy engine, on both challenge sets. Needs a GPU."""
    from challenge_set import generate_challenge_set
    from challenge_set_v2 import generate_challenge_set_v2

    model, tok = load_extractor()
    results = {}
    for name, df in [("challenge_v1", generate_challenge_set()),
                     ("challenge_v2", generate_challenge_set_v2())]:
        preds = [classify_hybrid(model, tok, r) for r in df["raw_log"]]
        out, acc = _score(df, preds, f"HYBRID {name}")
        out.to_csv(f"hybrid_results_{name}.csv", index=False)
        results[name] = acc

    print("\n" + "=" * 62)
    print("ARCHITECTURE COMPARISON  (risk accuracy)")
    print("=" * 62)
    print(f"{'approach':<34}{'chal v1':>10}{'chal v2':>12}")
    print(f"{'always-Low (trivial)':<34}{'38.5%':>10}{'36.0%':>12}")
    print(f"{'fine-tuned LLM (Session 4)':<34}{'89.7%':>10}{'48.0%':>12}")
    print(f"{'regex extract + policy':<34}{'64.1%':>10}{'80.0%':>12}")
    print(f"{'LLM extract + policy (hybrid)':<34}"
          f"{results['challenge_v1']:>9.1f}%{results['challenge_v2']:>11.1f}%")
    return results


def run_regex_eval():
    """No-GPU variant: regex extractor + policy engine (see eval_pipeline.py)."""
    from audit_policy import classify
    from challenge_set_v2 import generate_challenge_set_v2
    df = generate_challenge_set_v2()
    preds = [classify(r) for r in df["raw_log"]]
    return _score(df, preds, "REGEX challenge_v2")


if __name__ == "__main__":
    run_regex_eval()
