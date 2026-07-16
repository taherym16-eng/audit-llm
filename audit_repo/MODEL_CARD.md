# Model Card — Enterprise IT Audit Log Risk Classification & PII Detection

**Status:** research prototype. **Not fit for production deployment.** See *Limitations*.

**Author's summary in one line:** a fine-tuned 3B model scored **100%** on its own
test set and **48%** on a held-out probe of novel log formats — this card documents
why, and what architecture fixed it.

---

## 1. What this is

A pipeline that reads a single raw enterprise audit log line (syslog, Apache,
JSON, CEF, CloudTrail, GCP, Okta, Azure AD, Kubernetes audit, Windows event,
auditd, Zeek, RFC5424, Sysmon, EDR) and emits:

| field | values |
|---|---|
| `risk_level` | Low / Medium / High / Critical |
| `policy_violation` | 0 / 1 |
| `unauthorized_pii_access` | 0 / 1 |
| `reason` | short justification naming the deciding cue |
| `needs_human_review` | 0 / 1 — set when any input fact is UNKNOWN |

Two implementations are provided and compared:

- **A. Fine-tuned LLM** (Sessions 1–4): QLoRA-tuned `Qwen2.5-3B-Instruct` reads the
  raw log and outputs the verdict directly.
- **B. Hybrid pipeline** (Session 5): LLM extracts *facts*; deterministic code looks
  up entitlement/classification/approval; a policy engine decides severity.

**B is the recommended design.** A is retained because its failure is the most
informative result in the project.

---

## 2. Data

All data is **synthetic**. No real logs, no real PII. Secrets appearing in examples
are public placeholders (e.g. AWS's documented `AKIAIOSFODNN7EXAMPLE`).

| set | n | origin | role |
|---|---|---|---|
| frozen test | 67 | seeded generator (seed 42) | in-distribution sanity anchor; never trained on |
| train / val (v2) | 880 / 120 | seeded generators, widened pools | fine-tuning |
| challenge v1 | 39 | **hand-written** | OOD probe (Session 3); became a *regression* set after Session 4 trained on its families |
| challenge v2 | 25 | **hand-written**, disjoint formats & scenarios | the honest OOD probe |

Anti-leakage is enforced programmatically: training rows are dropped if their
`raw_log` appears in *any* held-out set (assertions in `make_data_v2.py`).

Challenge sets are labelled by novelty type — `format_novel`, `fp_trap`
(dangerous-looking but authorized), `adversarial` (real breach, disguised),
`semantic_novel`, `judgment_call` — so failures can be attributed to a mechanism
rather than averaged away.

---

## 3. Results

### 3.1 The headline

| set | zero-shot | fine-tuned v1 | fine-tuned v2 | **hybrid pipeline** | *always-Low* |
|---|---|---|---|---|---|
| frozen test (67) | 68.7% | 100.0% | 100.0% | — | 41.8% |
| challenge v1 (39) | — | 53.8% | 89.7% | 64.1%¹ | 38.5% |
| **challenge v2 (25)** | — | — | **48.0%** | **80.0%**¹ | **36.0%** |

¹ regex-extractor variant, measured. The LLM-extractor variant (`hybrid_pipeline.run_hybrid_eval()`)
requires a GPU and is **not yet measured** — see *Open questions*.

**Always-Low is the number that matters.** A model that always predicts "Low"
scores 36.0% on challenge v2. The fine-tuned model scored 48.0% — a real lift of
**+12 points, roughly 3 rows out of 25**.

### 3.2 Per-slice, challenge v2 (the honest probe)

| slice | n | always-Low | fine-tuned v2 | hybrid |
|---|---|---|---|---|
| adversarial (real breaches, disguised) | 7 | 0.0% | **14.3%** | **100.0%** |
| format_novel | 6 | 50.0% | **83.3%** | 66.7% |
| fp_trap (authorized, looks dangerous) | 7 | 85.7% | 85.7% | 85.7% |
| semantic_novel | 3 | 0.0% | 0.0% | 66.7% |
| judgment_call | 2 | 0.0% | 0.0% | 50.0% |

**Read the `fp_trap` row carefully.** Session 4 reported this slice improving from
25% → 85.7% and called it a win. But 6 of those 7 rows are labelled Low, so
*always predicting Low* also scores 85.7%. The fine-tuned model's fp_trap score is
**statistically indistinguishable from having acquired a Low prior** — which is
corroborated by its error direction: **11 under-escalations vs 2 over-escalations**
(mean severity delta −0.68) on challenge v2.

The hybrid scores the same 85.7% but by a different mechanism: it correctly calls
break-glass access **Medium** (the one non-Low trap) and its only miss is a benign
row it **flagged for human review**. Its prediction spread on challenge v2 is
Low 8 / Medium 7 / Critical 7 / High 3 — not a collapsed prior.

### 3.3 Unauthorized-PII flag, challenge v2

| | caught | missed | false alarms |
|---|---|---|---|
| fine-tuned v2 | 1 | **1** | 0 |
| hybrid | 2 | **0** | 0 |

n=2 positives. Directional only.

---

## 4. The four-iteration arc (what actually happened)

1. **Baseline.** Zero-shot `Qwen2.5-3B-Instruct`: 68.7%, 100% JSON parse rate.
2. **Fine-tune.** QLoRA on 704 leakage-free rows → **100.0%** on the frozen test set.
   *Treated as a red flag, not a success* — train and test shared a template generator.
3. **Challenge.** A hand-written OOD probe collapsed it to **53.8%**, with `fp_trap`
   at **25%**. Diagnosis: keyword overfitting (`sensitive table → Critical`) and
   outcome insensitivity (attempted vs achieved conflated).
4. **Remediate.** Contrastive data (authorized hard negatives with role→resource
   matching; attempted-vs-achieved pairs) + a rules-based prompt → challenge v1
   **89.7%**. But the fresh probe (challenge v2) came in at **48.0%**.

### 4.1 The finding

The Session-4 fix did not teach authorization reasoning. It **shifted the model's
prior toward Low**, which scores well on slices where Low is the answer and
collapses where it isn't. A pre-registered hypothesis — that deceptive *approval
tokens* (`granted:true`, `APPROVED`) were dragging predictions down — was **tested
and refuted**: rows containing approval tokens scored *better* (62.5% vs 41.2%).

The real mechanism is visible in the adversarial misses. Training defined "Critical"
via three attack families (PII table names; brute-force-then-success;
`AttachUserPolicy AdministratorAccess`). **Every adversarial miss was a Critical
event outside that list** — ransomware, audit-log tampering, OAuth token replay,
service-principal backdoor, CEF-format exfiltration. Two misses are decisive:

- `sales_lee` uploading 5.3 GB of `customers_full.csv` to an external host → predicted **Low**
- `growth_bot` deleting the GCP logging sink → predicted **Low**

Both actors are *in the training data as unauthorized principals*. The model failed
because the **format** was unfamiliar, not the semantics.

> **Fine-tuning a small model on synthetic templates yields a closed-world
> classifier: it learns "these N patterns are breaches," not "a breach is an
> illegitimate actor achieving a harmful outcome." Attack space is open-ended, so
> no amount of additional attack families closes the gap. This is a structural
> ceiling, not a data-volume problem.**

Corollary: the obvious next fix — add ransomware and token-replay to training —
would teach those two signatures and fail on the next novel attack. We did not do it.

### 4.2 What the fine-tuned model *is* good at

One capability generalized cleanly: **reading formats it had never seen**
(`format_novel`: 83.3% vs 50.0% trivial, **+33.3**). The Session-5 architecture is
built on this measured strength.

---

## 5. Recommended architecture (Session 5)

```
raw log
   ↓
[ LLM extractor ]          audit_prompt_v3 — "what happened?", not "how bad is it?"
   ↓                       base model, no fine-tuning required
{actor, resource, action, outcome, volume, destination, approval_ref}
   ↓
[ deterministic lookups ]  audit_enrich — identity directory, data catalog,
   ↓                       entitlement matrix, change-management validation
{actor_role, resource_class, is_entitled, approval_valid}
   ↓
[ policy engine ]          audit_policy — ordered, auditable rules
   ↓
risk_level + flags + reason  (+ needs_human_review when any fact is UNKNOWN)
```

Three properties the fine-tuned model cannot offer:

1. **Legitimacy is looked up, not inferred.** The decisive contrast — `dbadmin`
   reading `hr_ssn` (Low) vs `intern_park` reading `payroll_ssn` (Critical) — differs
   by exactly one field, `is_entitled`, which comes from an RBAC lookup. Both logs
   contain `granted:true`. No amount of text reasoning makes this reliable; a lookup
   makes it trivial.
2. **Policy is auditable.** `is_entitled=False AND resource_class=PII_* AND
   outcome=SUCCESS → Critical` can be shown to a regulator, unit-tested, and
   versioned. "The model felt it was Critical" cannot.
3. **It fails loud.** Unknown action / unknown actor / unknown outcome →
   `needs_human_review=1`, not a silent Low. The measured failure of Sessions 1–4
   was *silent* under-escalation.

---

## 6. Limitations

**Methodological (read before quoting any number):**

- **The policy rules and the challenge labels share an author.** The hybrid's 80.0%
  partly measures self-consistency, not independent ground truth. Rules were written
  as general policy (entitlement × sensitivity × outcome × destination), never tuned
  per row — but this needs validation against labels from an independent auditor or
  real logs. **Treat 80.0% as a design demonstration, not a benchmark.**
- **Sample sizes are small.** Challenge v2 is 25 rows; slices are n=2–7. Only the
  error *direction* (11 vs 2) is a robust signal. `semantic_novel` (n=3) and
  `judgment_call` (n=2) at 0% are **noise and are not findings**.
- **`judgment_call` rows are contestable by construction** — labelled as cases where
  reasonable auditors disagree. Scoring against them measures agreement with the
  author's opinion.
- **All data is synthetic.** Generalization to production logs is unmeasured.
- **`eval_loss` reached 6.0e-05** during v2 training. This does not indicate quality;
  the validation set shares generators with train, so it measures memorization. It is
  useless as an early-stopping signal — a repeat of the Session-2 lesson.

**Functional:**

- **Single-log scope.** No session/time correlation, so impossible-travel, low-and-slow
  campaigns, and multi-event attack chains are out of reach by construction.
- **Action semantics is the residual open-world surface.** Entitlement and outcome are
  closed-world lookups; mapping a novel method name to an action class is not. Unknown
  actions route to human review rather than a guess.
- **The directory/entitlement matrix is mocked.** Production requires live LDAP/Okta/IAM
  and data-catalog integration; a stale directory silently produces wrong `is_entitled`.
- **Approval validation is a regex.** A real system must check the ticket is open,
  covers this actor, and is within its window. Citing a ticket number is not proof.
- **Regex extraction is brittle** (hybrid v1: 64.1%, with 18/39 rows flagged for review).
  This is precisely why the LLM extractor exists.

---

## 7. Intended use / out of scope

**Intended:** research and demonstration of evaluation methodology for LLM-based audit
triage; a worked example of diagnosing template-matching.

**Out of scope:** autonomous enforcement, adverse employment/legal action, sole-source
compliance evidence, or any deployment where a missed Critical carries real
consequence. Measured Critical recall on novel attacks for the fine-tuned model was
**1 in 7**.

---

## 8. Reproducing

```bash
python make_data_v2.py       # deterministic: 880/120, verified md5-stable
python eval_pipeline.py      # regex extractor + policy, no GPU
python analyze_v2.py         # error-direction diagnostic on saved results
# GPU:
python -c "import session4_cells; session4_cells.run_retraining(); session4_cells.run_eval()"
python -c "import hybrid_pipeline; hybrid_pipeline.run_hybrid_eval()"
```

**Environment:** Google Colab, T4 (16 GB). Qwen2.5 config defaults to bf16; the T4
has no bf16 GradScaler kernel. Load with `dtype=torch.float16`, keep LoRA params in
fp32, and re-sanitize *after* `SFTTrainer.__init__` (TRL re-casts adapters from
`config.torch_dtype` during construction). See `session4_cells.py`.

---

## 9. Open questions

1. **Does the LLM extractor beat the regex extractor?** The hypothesis — that
   `format_novel`'s +33.3 lift transfers to extraction and lifts hybrid v1 above 64.1%
   — is **untested**. This is the single highest-value next experiment.
2. **Would an independent auditor's labels reproduce the hybrid's 80.0%?** Unknown, and
   the most important external validation.
3. **Does a larger model (7B–14B) escape the closed-world ceiling, or just widen the
   memorized list?** Untested. The prior here is "widen the list."
4. **What is the review-flag rate on production logs?** 12/25 flagged on challenge v2
   would be operationally expensive; the precision/recall of the flag itself is unmeasured.
