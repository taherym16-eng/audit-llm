# ==========================================================================
#  colab_bootstrap.py  —  RUN THIS FIRST, EVERY SESSION
#  Makes the project survive runtime death by keeping everything on Drive.
#
#  Colab wipes /content when the runtime ends. Google Drive is NOT wiped.
#  So: source + adapters + results live on Drive; /content is scratch only.
# ==========================================================================

# ------------------------------------------------------------------ #
# %% CELL A1 — mount Drive and set up the project folder
# ------------------------------------------------------------------ #
from google.colab import drive
drive.mount('/content/drive')

import os, shutil

PROJ = '/content/drive/MyDrive/audit_llm'      # <- survives runtime death
os.makedirs(PROJ, exist_ok=True)
os.makedirs(f'{PROJ}/adapters', exist_ok=True)
os.makedirs(f'{PROJ}/results', exist_ok=True)
print('project dir:', PROJ)

# ------------------------------------------------------------------ #
# %% CELL A2 — put the source files on Drive (ONE TIME)
#   Upload the 11 .py files from the chat into /content, then run this once.
#   After this, you never upload again — they live on Drive.
# ------------------------------------------------------------------ #
SRC = ["audit_dataset.py", "audit_dataset_v2.py", "audit_eval_harness.py",
       "audit_prompt_v2.py", "make_data.py", "make_data_v2.py",
       "challenge_set.py", "challenge_set_v2.py", "session4_cells.py",
       "train_qlora.py", "eval_finetuned.py"]

# from google.colab import files; files.upload()      # <- uncomment for the one-time upload
for f in SRC:
    if os.path.exists(f'/content/{f}'):
        shutil.copy(f'/content/{f}', f'{PROJ}/{f}')
print('on Drive:', sorted(os.listdir(PROJ)))

# ------------------------------------------------------------------ #
# %% CELL A3 — EVERY SESSION: pull source from Drive into the working dir
# ------------------------------------------------------------------ #
import sys
for f in SRC:
    p = f'{PROJ}/{f}'
    if os.path.exists(p):
        shutil.copy(p, f'/content/{f}')
if '/content' not in sys.path:
    sys.path.insert(0, '/content')
print('restored to /content:', [f for f in SRC if os.path.exists(f'/content/{f}')])

# ------------------------------------------------------------------ #
# %% CELL A4 — install deps (needed every session; ~2 min)
# ------------------------------------------------------------------ #
# !pip install -q -U "transformers>=4.51" "trl>=0.21" "peft>=0.14" \
#     "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.45"

# ------------------------------------------------------------------ #
# %% CELL A5 — rebuild data (deterministic: reproduces 880/120 exactly)
# ------------------------------------------------------------------ #
import make_data_v2
make_data_v2.main()

# ------------------------------------------------------------------ #
# %% CELL A6 — train, then IMMEDIATELY copy the adapter to Drive
#   The adapter is the ONLY artifact that costs GPU time. Save it first,
#   celebrate second.
# ------------------------------------------------------------------ #
import session4_cells
session4_cells.run_retraining()

shutil.copytree('audit-qlora-adapter-v2',
                f'{PROJ}/adapters/audit-qlora-adapter-v2',
                dirs_exist_ok=True)
print('ADAPTER SAFE ON DRIVE ->', f'{PROJ}/adapters/audit-qlora-adapter-v2')

# ------------------------------------------------------------------ #
# %% CELL A7 — evaluate, then copy results to Drive
# ------------------------------------------------------------------ #
session4_cells.run_eval()

import glob
for f in glob.glob('v2_results_*.csv'):
    shutil.copy(f, f'{PROJ}/results/{f}')
print('results saved ->', f'{PROJ}/results')

# ------------------------------------------------------------------ #
# %% CELL A8 — if the adapter is ALREADY on Drive, skip training entirely
#   Use this to resume: restores the adapter and jumps straight to eval.
# ------------------------------------------------------------------ #
# saved = f'{PROJ}/adapters/audit-qlora-adapter-v2'
# if os.path.exists(saved):
#     shutil.copytree(saved, 'audit-qlora-adapter-v2', dirs_exist_ok=True)
#     print('adapter restored from Drive — no retraining needed')
#     import session4_cells; session4_cells.run_eval()
