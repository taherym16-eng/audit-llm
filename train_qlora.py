# ==========================================================================
#  train_qlora.py  —  Session 2: QLoRA fine-tuning on a Colab T4
#  Runtime -> Change runtime type -> T4 GPU. Run cells in order.
# ==========================================================================

# ------------------------------------------------------------------ #
# %% CELL 1 — install (modern TRL API: SFTConfig owns the data args)
# ------------------------------------------------------------------ #
# !pip install -q -U "transformers>=4.51" "trl>=0.21" "peft>=0.14" \
#     "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.45"

# ------------------------------------------------------------------ #
# %% CELL 2 — build the leakage-free splits
#   Needs audit_dataset.py, audit_eval_harness.py, make_data.py in the file panel.
# ------------------------------------------------------------------ #
import make_data
make_data.main()          # writes train.jsonl / val.jsonl / test.csv

# ------------------------------------------------------------------ #
# %% CELL 3 — load Qwen2.5-3B in 4-bit (QLoRA) on the T4
#   T4 notes: Turing GPU -> use fp16 (NOT bf16) and NO FlashAttention-2.
# ------------------------------------------------------------------ #
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "audit-qlora-adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,       # fp16 on T4, not bf16
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"                       # right padding for SFT

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map={"": 0},
    attn_implementation="sdpa",                  # T4-safe (flash-attn-2 unsupported)
)
model.config.use_cache = False                   # required with gradient checkpointing
print("loaded", MODEL_ID, "in 4-bit")

# ------------------------------------------------------------------ #
# %% CELL 4 — LoRA config + load datasets
# ------------------------------------------------------------------ #
from datasets import load_dataset
from peft import LoraConfig

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # all linear projections in a Qwen2.5 decoder block:
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

train_ds = load_dataset("json", data_files="train.jsonl", split="train")
val_ds = load_dataset("json", data_files="val.jsonl", split="train")
print(train_ds, val_ds)

# ------------------------------------------------------------------ #
# %% CELL 5 — SFTConfig + SFTTrainer, then train
#   prompt/completion data -> completion_only_loss is applied automatically
#   (the model is supervised on the JSON only, not on the long system prompt).
# ------------------------------------------------------------------ #
from trl import SFTConfig, SFTTrainer

cfg = SFTConfig(
    output_dir="sft_out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,               # effective batch 16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=768,                              # covers system prompt + log + JSON
    packing=False,
    completion_only_loss=True,                   # loss on the JSON completion only
    optim="paged_adamw_8bit",
    fp16=True,                                   # T4: fp16 on
    bf16=False,                                  # T4: bf16 off
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
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tok,                         # modern TRL (was `tokenizer=`)
    peft_config=lora,                            # TRL applies LoRA + kbit prep
)
trainer.train()

trainer.save_model(ADAPTER_DIR)                  # saves the LoRA adapter only
tok.save_pretrained(ADAPTER_DIR)
print("adapter saved ->", ADAPTER_DIR)

# ------------------------------------------------------------------ #
# %% CELL 6 — quick eyeball on 3 held-out test logs
# ------------------------------------------------------------------ #
import pandas as pd
from audit_eval_harness import build_messages

model.config.use_cache = True
sample = pd.read_csv("test.csv").sample(3, random_state=1)
for _, row in sample.iterrows():
    msgs = build_messages(row["raw_log"])
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print("LOG  :", row["raw_log"][:90])
    print("TRUE :", row["risk_level"], "| pred:", gen.strip()[:120], "\n")
