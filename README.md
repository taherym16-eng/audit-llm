# Enterprise IT Audit Log Risk Classification & PII Detection


> 🛡️ **Project Lens:** This repository is an adversarial audit of LLMs applied to security compliance. Written from an IT Auditor's perspective, it demonstrates how to identify "paper-thin compliance" (where models learn keyword shortcuts to pass evaluations) and how to design deterministic, policy-enforced guardrails to guarantee 100% verification on critical security events.


Classifying raw enterprise audit logs (Low/Medium/High/Critical + policy violation +
unauthorized PII access) with a small open-source LLM — and a case study in why the
headline accuracy number was lying.

> **TL;DR** — A QLoRA fine-tune of `Qwen2.5-3B-Instruct` scored **100.0%** on its own
> held-out test set. On 25 hand-written logs in formats it had never seen, it scored
> **48.0%** — against a **36.0%** floor for a model that just always says "Low". This
> repo documents the diagnosis and the architecture that fixed it (**80.0%**, and
> **100%** on the disguised-breach slice where the fine-tune got **1 of 7**).

---

## The arc

| # | question | result |
|---|---|---|
| 1 | Can a 3B model triage audit logs zero-shot? | 68.7%, 100% JSON parse rate |
| 2 | Does QLoRA fine-tuning help? | **100.0%** — treated as a red flag, not a win |
| 3 | Does it generalize to novel formats? | **53.8%**; false-positive traps at **25%** |
| 4 | Does contrastive data fix it? | challenge v1 → 89.7%… but a *fresh* probe → **48.0%** |
| 5 | Why? | It learned a **closed list of breach signatures**, and shifted its prior to Low |
| 5 | What actually works? | **LLM extracts facts → code looks up entitlement → policy engine decides** |

Full numbers, per-slice breakdowns, refuted hypotheses, and every caveat:
**[MODEL_CARD.md](MODEL_CARD.md)**.

## The finding

Training defined "Critical" through three attack families. Every disguised-breach miss
was an attack *outside that list* — ransomware, audit-log tampering, OAuth token replay.
Two of them (`sales_lee` exfiltrating 5.3 GB of customer data; `growth_bot` deleting the
audit log sink) involve actors that **are in the training data as unauthorized
principals**. The model said **Low** to both, because the *format* was unfamiliar.

> Fine-tuning a small model on synthetic templates produces a closed-world classifier.
> It learns "these N patterns are breaches," not "a breach is an illegitimate actor
> achieving a harmful outcome." Attack space is open-ended — so adding more attack
> families cannot close the gap. It's a structural ceiling, not a data-volume problem.

**The one thing that did generalize:** reading unseen log formats (+33.3 points over
trivial). So Session 5 uses the LLM for *exactly that* and nothing else.

## The architecture

```
raw log → [ LLM extractor ] → {actor, resource, action, outcome, volume, destination}
                                              ↓
                              [ directory / catalog / entitlement lookups ]
                                              ↓
                              {actor_role, resource_class, is_entitled}
                                              ↓
                              [ policy engine ] → risk + flags + reason
                                              ↓
                              unknown fact? → needs_human_review (fail loud)
```

The decisive case: `dbadmin` reading `hr_ssn` (**Low**) vs `intern_park` reading
`payroll_ssn` (**Critical**). Both logs contain `granted:true`. Both are a successful
read of HR PII. They differ by **one field** — `is_entitled` — which is an RBAC lookup,
not something to infer from text.

## Files

**Pipeline (Session 5, recommended)**
| file | role |
|---|---|
| `audit_enrich.py` | identity directory, data catalog, entitlement matrix, approval validation |
| `audit_policy.py` | deterministic severity rules over enriched facts |
| `audit_prompt_v3.py` | LLM-as-extractor prompt (+ few-shot) |
| `hybrid_pipeline.py` | LLM extract → lookups → policy, end to end |
| `eval_pipeline.py` | scores the pipeline vs the fine-tune vs always-Low |

**Fine-tuning track (Sessions 1–4)**
| file | role |
|---|---|
| `audit_dataset.py` / `audit_dataset_v2.py` | seeded synthetic generators (v2 adds contrastive families) |
| `make_data.py` / `make_data_v2.py` | leakage-free splits (assertions enforce disjointness) |
| `audit_prompt_v2.py` | rules-based severity prompt |
| `session4_cells.py` | QLoRA training + eval (T4-hardened; see bf16 notes) |
| `audit_eval_harness.py` | prompt → generate → parse → metrics |

**Evaluation**
| file | role |
|---|---|
| `challenge_set.py` | 39 hand-written OOD logs (now a regression set) |
| `challenge_set_v2.py` | 25 hand-written logs, disjoint formats/scenarios — the honest probe |
| `analyze_v2.py` | error-direction + hypothesis diagnostics |
| `colab_bootstrap.py` | Drive persistence (Colab wipes `/content` on disconnect) |

## Quickstart

```bash
pip install pandas scikit-learn
python eval_pipeline.py        # no GPU: pipeline vs fine-tune vs trivial baseline
```

GPU (Colab T4): run `colab_bootstrap.py` cells, then `session4_cells.run_retraining()`
and `hybrid_pipeline.run_hybrid_eval()`.

## Methodology notes worth stealing

- **Always compute the trivial baseline.** "Always predict the majority class" scored
  85.7% on the slice we were celebrating at 85.7%. Without it, we'd have shipped a prior
  shift as a reasoning win.
- **A test set from the same generator as training measures memorization.** Hand-write
  an OOD probe. When you train on its failure modes, it becomes a *regression* set —
  write a new probe.
- **Pre-register the hypothesis, then try to kill it.** Ours ("approval tokens drag
  predictions to Low") was refuted by its own diagnostic. The refutation is in the model
  card.
- **100% is a bug report.** So is `eval_loss = 6e-05`.
- **Fail loud.** The worst measured failure wasn't a wrong answer — it was a *confident
  silent* Low on a 5.3 GB exfiltration.

## Caveats

All data synthetic; no real logs or PII. The policy rules and challenge labels share an
author, so the hybrid's 80.0% partly measures self-consistency — it's a design
demonstration, not a benchmark. Slices are n=2–7; only error *direction* is robust.
Research prototype: **not fit for production**. See
[MODEL_CARD.md §6](MODEL_CARD.md) for the full list.
