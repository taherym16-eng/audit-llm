# ==========================================================================
#  session4_cells.py  —  Session 4, T4-HARDENED (bf16-proof) + function API
#
#  Two separate bf16 traps on a Colab T4, both handled here:
#    (1) base model: Qwen2.5 config defaults to bf16 for un-quantized layers.
#        -> load with dtype=torch.float16.
#    (2) LoRA ADAPTERS (the trainable params): SFTTrainer.__init__ reads
#        model.config.torch_dtype (Qwen2.5 hardcodes bfloat16) and RE-CASTS the
#        adapters to bf16 during construction -- even ones we made fp32, even with
#        no peft_config. bf16 grads then hit the fp16 GradScaler's unscale_, which
#        has no bf16 CUDA kernel -> "NotImplementedError ... for 'BFloat16'".
#        -> (a) overwrite config.torch_dtype = fp16 so the derivation lands on
#               fp16, and (b) re-sanitize trainer.model to fp32 AFTER construction
#               (the last point before train(); nothing re-casts params after).
#        fp32 trainable + fp16 GradScaler is fine: unscale_ supports fp32 and fp16,
#        only bf16 is unimplemented.
#
#  Trainable params in fp32 + fp16 autocast + GradScaler is the standard, stable
#  QLoRA recipe: the scaler unscales fp32 grads fine (fp32 IS supported; only
#  bf16 is not). We pass the wrapped model WITHOUT peft_config so the trainer
#  does not re-create adapters.
#
#  Usage in Colab:
#     import session4_cells
#     session4_cells.run_retraining()     # builds data if missing, trains, saves
#     session4_cells.run_eval()           # three-way scoreboard
#  Files needed in panel: audit_dataset.py, audit_dataset_v2.py,
#  audit_eval_harness.py, audit_prompt_v2.py, make_data.py, make_data_v2.py,
#  challenge_set.py, challenge_set_v2.py.   Runtime -> T4 GPU.
# ==========================================================================

import os
import gc

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import audit_eval_harness as H
from audit_prompt_v2 import build_messages_v2

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_V2 = "audit-qlora-adapter-v2"


# --------------------------------------------------------------------------- #
# dtype guard: no bf16 anywhere; trainable params in fp32 (scaler-safe)
# --------------------------------------------------------------------------- #
def sanitize_dtypes(model):
    """Trainable params -> fp32; any bf16 (frozen or buffer) -> fp16. Never
    touches 4-bit quantized weights (those are uint8, not a float dtype)."""
    changed = 0
    for _, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32); changed += 1          # LoRA -> fp32
        elif (not p.requires_grad) and p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float16); changed += 1          # frozen bf16 -> fp16
    for b in model.buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.to(torch.float16); changed += 1
    return changed


def _assert_no_bf16(model, where=""):
    train_dtypes = {p.dtype for _, p in model.named_parameters() if p.requires_grad}
    assert torch.bfloat16 not in train_dtypes, f"bf16 in trainable params {where}: {train_dtypes}"
    return train_dtypes


def _load_4bit():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,       # fp16 compute (not bf16)
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        dtype=torch.float16,                        # modern spelling of torch_dtype (no warning)
        device_map={"": 0},
        attn_implementation="sdpa",                 # T4: no flash-attn-2
    )
    model.config.use_cache = False
    # Qwen2.5's config hardcodes torch_dtype=bfloat16; TRL/PEFT read this to decide
    # adapter dtype and will re-cast LoRA to bf16. Overwrite it so any dtype
    # derivation lands on fp16 (scaler-safe) instead of bf16.
    model.config.torch_dtype = torch.float16
    try:
        model.config.dtype = torch.float16
    except (AttributeError, TypeError):
        pass                                        # some versions expose dtype read-only
    return model, tok


# --------------------------------------------------------------------------- #
def build_data():
    import make_data_v2
    make_data_v2.main()


def run_diagnostic():
    """Base model + v2 prompt (no fine-tuning) on challenge v1. Optional."""
    from challenge_set import generate_challenge_set
    H.build_messages = build_messages_v2
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map={"": 0})
    m = H.score(H.run_baseline(generate_challenge_set(), base, tok, verbose=False))
    print(f"\nDIAGNOSTIC base + v2 prompt on challenge v1: {m['risk_accuracy']*100:.1f}%")
    del base; gc.collect(); torch.cuda.empty_cache()
    return m


# --------------------------------------------------------------------------- #
def run_retraining():
    if not os.path.exists("train_v2.jsonl"):
        build_data()

    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_4bit()

    # explicit PEFT build so WE control adapter creation + dtype
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    n = sanitize_dtypes(model)                       # <-- the real fix
    dt = _assert_no_bf16(model, "after get_peft_model")
    print(f"sanitize_dtypes: fixed {n} tensors | trainable dtypes = {dt}")
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files="train_v2.jsonl", split="train")
    val_ds = load_dataset("json", data_files="val_v2.jsonl", split="train")

    cfg = SFTConfig(
        output_dir="sft_out_v2",
        num_train_epochs=3,
        per_device_train_batch_size=4,               # OOM? -> 2, and grad_accum -> 8
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        # gradient checkpointing already enabled via prepare_model_for_kbit_training,
        # so we do NOT re-enable it here (avoids double-enable).
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=1024,
        packing=False,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        fp16=True,                                   # T4 mixed precision ON
        bf16=False,                                  # bf16 OFF (declared once)
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,                        # NOTE: no peft_config (model is already PEFT)
    )
    # SFTTrainer.__init__ re-casts adapters to config.torch_dtype. Re-sanitize the
    # trainer's model HERE — this is the last point before train(), and nothing
    # re-casts params afterward (fp16 autocast doesn't touch stored param dtype).
    fixed = sanitize_dtypes(trainer.model)
    dt = _assert_no_bf16(trainer.model, "after SFTTrainer init (post-sanitize)")
    print(f"post-init sanitize: fixed {fixed} tensors | trainable dtypes = {dt}")

    trainer.train()
    trainer.save_model(ADAPTER_V2)
    tok.save_pretrained(ADAPTER_V2)
    print("saved ->", ADAPTER_V2)

    del model, trainer
    gc.collect(); torch.cuda.empty_cache()
    return ADAPTER_V2


# --------------------------------------------------------------------------- #
def run_eval():
    from peft import PeftModel
    from challenge_set import generate_challenge_set
    from challenge_set_v2 import generate_challenge_set_v2

    H.build_messages = build_messages_v2             # eval under the v2 prompt
    tok = AutoTokenizer.from_pretrained(ADAPTER_V2)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map={"": 0})
    ft = PeftModel.from_pretrained(base, ADAPTER_V2)
    ft.eval()

    def evaluate(df, name):
        res = H.run_baseline(df, ft, tok, verbose=False)
        m = H.score(res)
        if "novelty_type" in res.columns:
            res["ok"] = res["risk_level"] == res["pred_risk_level"]
            g = res.groupby("novelty_type")["ok"].agg(["mean", "count"])
            g["mean"] = (g["mean"] * 100).round(1)
            print(f"\n[{name}] accuracy by novelty_type:")
            print(g.rename(columns={"mean": "acc_%", "count": "n"}).to_string())
        res.to_csv(f"v2_results_{name}.csv", index=False)
        return m

    m_test = evaluate(pd.read_csv("test.csv"), "frozen_test")
    m_c1 = evaluate(generate_challenge_set(), "challenge_v1")
    m_c2 = evaluate(generate_challenge_set_v2(), "challenge_v2")

    print("\n" + "=" * 64)
    print("SESSION 4 SCOREBOARD  (risk accuracy)")
    print("=" * 64)
    print(f"{'set':<24}{'v1 adapter':>13}{'v2 adapter':>13}")
    print(f"{'frozen test':<24}{'100.0%':>13}{m_test['risk_accuracy']*100:>12.1f}%")
    print(f"{'challenge v1 (regr.)':<24}{'53.8%':>13}{m_c1['risk_accuracy']*100:>12.1f}%")
    print(f"{'challenge v2 (NEW OOD)':<24}{'—':>13}{m_c2['risk_accuracy']*100:>12.1f}%")
    print("\nchallenge v1 = regression (are the fixed bugs fixed?).")
    print("challenge v2 = the honest generalization number.")
    return m_test, m_c1, m_c2


if __name__ == "__main__":
    build_data()
    run_retraining()
    run_eval()
