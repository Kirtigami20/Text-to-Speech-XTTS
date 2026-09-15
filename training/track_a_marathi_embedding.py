"""
TRACK A — Custom Marathi Language Embedding Inspector & Generator
================================================================
Findings (discovered by runtime inspection, not assumed):
  - Language conditioning in XTTS-v2 is implemented via special tokens in the
    text tokenizer, NOT a separate language-embedding module.
  - Token ID [hi] = 6680  (last slot in the 6681-token vocab)
  - The text token embedding table: gpt.text_embedding.weight  shape=(6681, 1024)
  - The text output projection:     gpt.text_head.weight        shape=(6681, 1024)
    (these are typically tied weights in GPT-style models)
  - To add [mr] we must:
      1. Append a new row (clone of [hi] row) to both tensors -> shape becomes (6682, 1024)
      2. Add [mr] as token id=6681 in vocab.json added_tokens
      3. Save the modified checkpoint and the modified vocab.json

This script does NOT train anything.
Run it ONLY to prepare the Track A checkpoint for a future, separately approved experiment.
"""

import sys
import copy
import json
import torch
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(r"C:\gray matrix\text_to_speech")
CHECKPOINTS_PATH = PROJECT_ROOT / "training" / "xtts_output" / "XTTS_v2_original_model_files"
CHECKPOINT_FILE  = CHECKPOINTS_PATH / "model.pth"
VOCAB_FILE       = CHECKPOINTS_PATH / "vocab.json"
OUTPUT_DIR       = PROJECT_ROOT / "training" / "xtts_experiment"
OUTPUT_CKPT      = OUTPUT_DIR / "xtts_v2_mr_extended.pth"
OUTPUT_VOCAB     = OUTPUT_DIR / "vocab_mr_extended.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Discovered keys (do not change without re-inspecting the state-dict) ──
EMB_KEY  = "gpt.text_embedding.weight"   # input embedding   (6681 × 1024)
HEAD_KEY = "gpt.text_head.weight"        # output projection (6681 × 1024)

HI_TOKEN_ID = 6680   # [hi] is the 6681st token (0-indexed row 6680)
MR_TOKEN_ID = 6681   # new [mr] will be appended as row 6681

print("=" * 70)
print("TRACK A — MARATHI LANGUAGE EMBEDDING GENERATOR")
print("=" * 70)

# ── 1. Load checkpoint ─────────────────────────────────────────────────────
print(f"\n[1] Loading checkpoint:\n    {CHECKPOINT_FILE}")
checkpoint = torch.load(str(CHECKPOINT_FILE), map_location="cpu", weights_only=False)

if "model" in checkpoint:
    state_dict = checkpoint["model"]
    wrapper_key = "model"
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    wrapper_key = "state_dict"
else:
    state_dict = checkpoint
    wrapper_key = None

print(f"    Top-level keys : {list(checkpoint.keys()) if isinstance(checkpoint, dict) else '<raw state-dict>'}")

# ── 2. Verify embedding tensor shapes ──────────────────────────────────────
print(f"\n[2] Verifying embedding tensors...")

assert EMB_KEY  in state_dict, f"Key '{EMB_KEY}' not found in state_dict!"
assert HEAD_KEY in state_dict, f"Key '{HEAD_KEY}' not found in state_dict!"

emb_tensor  = state_dict[EMB_KEY]   # (6681, 1024)
head_tensor = state_dict[HEAD_KEY]  # (6681, 1024)

print(f"    {EMB_KEY}  : {tuple(emb_tensor.shape)}")
print(f"    {HEAD_KEY} : {tuple(head_tensor.shape)}")

assert emb_tensor.shape[0]  == 6681, f"Unexpected emb shape: {emb_tensor.shape}"
assert head_tensor.shape[0] == 6681, f"Unexpected head shape: {head_tensor.shape}"
assert emb_tensor.shape == head_tensor.shape

# ── 3. Extract the [hi] row ────────────────────────────────────────────────
print(f"\n[3] Extracting [hi] row (index {HI_TOKEN_ID})...")

hi_emb_row  = emb_tensor[HI_TOKEN_ID].clone()
hi_head_row = head_tensor[HI_TOKEN_ID].clone()

print(f"    [hi] emb row  — norm: {hi_emb_row.norm().item():.4f}  first 6 vals: {hi_emb_row[:6].tolist()}")
print(f"    [hi] head row — norm: {hi_head_row.norm().item():.4f}  first 6 vals: {hi_head_row[:6].tolist()}")

# ── 4. Create [mr] row (clone of [hi]) ────────────────────────────────────
print(f"\n[4] Creating [mr] row as clone of [hi]...")

mr_emb_row  = hi_emb_row.clone()
mr_head_row = hi_head_row.clone()

print(f"    [mr] emb row  — norm: {mr_emb_row.norm().item():.4f}  first 6 vals: {mr_emb_row[:6].tolist()}")
print(f"    [mr] head row — norm: {mr_head_row.norm().item():.4f}  first 6 vals: {mr_head_row[:6].tolist()}")

# ── 5. Extend tensors ──────────────────────────────────────────────────────
print(f"\n[5] Extending embedding tables from 6681 → 6682 rows...")

new_emb_tensor  = torch.cat([emb_tensor,  mr_emb_row.unsqueeze(0)],  dim=0)
new_head_tensor = torch.cat([head_tensor, mr_head_row.unsqueeze(0)], dim=0)

print(f"    Original {EMB_KEY}  : {tuple(emb_tensor.shape)}")
print(f"    Modified {EMB_KEY}  : {tuple(new_emb_tensor.shape)}")
print(f"    Original {HEAD_KEY} : {tuple(head_tensor.shape)}")
print(f"    Modified {HEAD_KEY} : {tuple(new_head_tensor.shape)}")

# ── 6. Build new state-dict ────────────────────────────────────────────────
new_state_dict = dict(state_dict)
new_state_dict[EMB_KEY]  = new_emb_tensor
new_state_dict[HEAD_KEY] = new_head_tensor

if wrapper_key:
    new_checkpoint = dict(checkpoint)
    new_checkpoint[wrapper_key] = new_state_dict
else:
    new_checkpoint = new_state_dict

# ── 7. Save extended checkpoint ───────────────────────────────────────────
print(f"\n[6] Saving extended checkpoint:\n    {OUTPUT_CKPT}")
torch.save(new_checkpoint, str(OUTPUT_CKPT))
size_mb = OUTPUT_CKPT.stat().st_size / (1024 * 1024)
print(f"    Saved ({size_mb:.0f} MB).")

# ── 8. Extend vocab.json ──────────────────────────────────────────────────
print(f"\n[7] Extending vocab.json with [mr] token (id={MR_TOKEN_ID})...")

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

# Check [mr] not already present
existing_contents = [t["content"] for t in vocab_data.get("added_tokens", [])]
if "[mr]" in existing_contents:
    print("    [mr] already exists in vocab. Skipping vocab modification.")
else:
    mr_token_entry = {
        "id": MR_TOKEN_ID,
        "special": True,
        "content": "[mr]",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False
    }
    vocab_data["added_tokens"].append(mr_token_entry)

    # Also update model vocab if present
    if "model" in vocab_data and "vocab" in vocab_data["model"]:
        vocab_data["model"]["vocab"]["[mr]"] = MR_TOKEN_ID

    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    size_kb = OUTPUT_VOCAB.stat().st_size / 1024
    print(f"    Saved extended vocab.json ({size_kb:.0f} KB):\n    {OUTPUT_VOCAB}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRACK A — COMPLETE (No training started)")
print("=" * 70)
print(f"""
  Exact state-dict keys modified:
    {EMB_KEY}   (6681,1024) → (6682,1024)
    {HEAD_KEY}  (6681,1024) → (6682,1024)

  Source row   : [hi]  (token id {HI_TOKEN_ID}, row index {HI_TOKEN_ID})
  New row      : [mr]  (token id {MR_TOKEN_ID}, row index {MR_TOKEN_ID})
  Initialization: clone of [hi] row

  Extended checkpoint : {OUTPUT_CKPT}
  Extended vocab.json : {OUTPUT_VOCAB}

  NEXT STEP (only after E1/E2 results reviewed and approved):
    - Use OUTPUT_CKPT as xtts_checkpoint in a new training config
    - Use OUTPUT_VOCAB as tokenizer_file
    - Marathi/Marnglish samples → language='mr'
    - Hindi/Hinglish samples    → language='hi'
    - English samples           → language='en'
    - Monitor Hindi val loss separately to confirm [hi] embedding is undisturbed
""")
