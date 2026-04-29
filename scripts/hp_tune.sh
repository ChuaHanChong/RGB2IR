#!/usr/bin/env bash
# hp_tune.sh — HP sweep for FLUX.2-Klein-4B RGB→IR LoRA training.
#
# Sweep design informed by 22-experiment optimization on grayscale→IR (Klein-4B):
#   - Prodigy optimizer beats AdamW by +9-12% PSNR  → fixed at Prodigy lr=1.0
#   - batch_size=1 only (OOM at 256x256)            → fixed
#   - guidance_scale=1.0 (no effect)                → fixed
#   - dropout >= 0.05 required for >100 epochs      → swept at 0.05 / 0.10
#   - rank 64 + FF layers ≈ rank 128 (no bottleneck)→ swept at 32 / 64 / 128
#   - cosmap weighting > sigma_sqrt > uniform       → swept
#   - FF layers help marginally (<0.2 dB PSNR)      → swept
#
# Each config: train → inference → eval. Skips already-completed configs.
# Outputs leaderboard at the end.
#
# Usage:
#   bash hp_tune.sh [GPU_ID]
#
# Edit DATASET / VAL_INPUT / VAL_GT / OUT_ROOT below before running.

set -uo pipefail

# -----------------------------------------------------------------------------
# Edit these paths
# -----------------------------------------------------------------------------
DATASET="/path/to/HFDataset_rgb2ir"
VAL_INPUT="/path/to/synthetic_rgb/test"
VAL_GT="/path/to/real_ir/test"
OUT_ROOT="/path/to/hp_sweep"
PROMPT="turn the visible image of Marine Vessel into sks infrared"

# Runtime
GPU="${1:-0}"
PYTHON="${PYTHON:-/data/hanchong/miniconda3/envs/flux/bin/python}"
ACCELERATE="${ACCELERATE:-/data/hanchong/miniconda3/envs/flux/bin/accelerate}"
PORT=29500

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME="${HF_HOME:-/data/hanchong/.cache/huggingface}"

mkdir -p "$OUT_ROOT/results" "$OUT_ROOT/logs" "$OUT_ROOT/loras" "$OUT_ROOT/gen"

# -----------------------------------------------------------------------------
# LoRA layer sets
# -----------------------------------------------------------------------------
ATTN_ONLY="to_k,to_q,to_v,to_out.0"
ATTN_PLUS_FF="to_k,to_q,to_v,to_out.0,to_qkv_mlp_proj,add_q_proj,add_k_proj,add_v_proj,to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2"

# -----------------------------------------------------------------------------
# Configurations to sweep
# Format: "name|overrides"
# Each entry inherits the base config (Prodigy, lr=1, batch=1, grad_accum=16, etc.)
# and applies its overrides on top.
# -----------------------------------------------------------------------------
declare -a CONFIGS=(
    # Baseline: best config from grayscale→IR optimization (exp-009/exp-015)
    "baseline|--rank 64 --lora_alpha 64 --lora_dropout 0.05 --num_train_epochs 150 --weighting_scheme cosmap --lora_layers $ATTN_PLUS_FF"

    # Rank ablation
    "rank32|--rank 32 --lora_alpha 32 --lora_dropout 0.05 --num_train_epochs 150 --weighting_scheme cosmap --lora_layers $ATTN_PLUS_FF"
    "rank128|--rank 128 --lora_alpha 128 --lora_dropout 0.05 --num_train_epochs 150 --weighting_scheme cosmap --lora_layers $ATTN_PLUS_FF"

    # Epochs ablation
    "ep100|--rank 64 --lora_alpha 64 --lora_dropout 0.05 --num_train_epochs 100 --weighting_scheme cosmap --lora_layers $ATTN_PLUS_FF"
    "ep200|--rank 64 --lora_alpha 64 --lora_dropout 0.05 --num_train_epochs 200 --weighting_scheme cosmap --lora_layers $ATTN_PLUS_FF"

    # Regularization ablation
    "dropout0.10|--rank 64 --lora_alpha 64 --lora_dropout 0.10 --num_train_epochs 150 --weighting_scheme cosmap --lora_layers $ATTN_PLUS_FF"

    # LoRA targets ablation
    "attn_only|--rank 64 --lora_alpha 64 --lora_dropout 0.05 --num_train_epochs 150 --weighting_scheme cosmap --lora_layers $ATTN_ONLY"

    # Weighting scheme ablation
    "weighting_sqrt|--rank 64 --lora_alpha 64 --lora_dropout 0.05 --num_train_epochs 150 --weighting_scheme sigma_sqrt --lora_layers $ATTN_PLUS_FF"
)

# -----------------------------------------------------------------------------
# Run sweep
# -----------------------------------------------------------------------------
for ENTRY in "${CONFIGS[@]}"; do
    NAME="${ENTRY%%|*}"
    OVERRIDES="${ENTRY#*|}"

    LORA_OUT="$OUT_ROOT/loras/$NAME"
    GEN_OUT="$OUT_ROOT/gen/$NAME"
    LOG="$OUT_ROOT/logs/$NAME.log"
    RESULT="$OUT_ROOT/results/$NAME.json"

    if [ -f "$RESULT" ]; then
        echo "[$(date)] $NAME — already done, skipping"
        continue
    fi

    echo "============================================================"
    echo "[$(date)] $NAME — starting (overrides: $OVERRIDES)"
    echo "============================================================"

    # ---- Train ----
    $ACCELERATE launch \
        --num_processes 1 --main_process_port $PORT \
        train_dreambooth_lora_flux2_klein_img2img.py \
        --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
        --dataset_name "$DATASET" \
        --image_column target_image \
        --cond_image_column cond_image \
        --instance_prompt "$PROMPT" \
        --output_dir "$LORA_OUT" \
        --resolution 256 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1.0 \
        --lr_scheduler cosine \
        --lr_warmup_steps 100 \
        --guidance_scale 1.0 \
        --optimizer prodigy \
        --mixed_precision bf16 \
        --gradient_checkpointing \
        --cache_latents \
        --seed 42 \
        --report_to none \
        --allow_tf32 \
        --random_flip \
        $OVERRIDES \
        2>&1 | tee "$LOG"

    # ---- Inference ----
    $PYTHON run_flux2.py \
        --lora_weights "$LORA_OUT" \
        --input_folder "$VAL_INPUT" \
        --output_folder "$GEN_OUT" \
        --model_name "black-forest-labs/FLUX.2-klein-4B" \
        --prompt "$PROMPT" \
        --guidance_scale 1.0 \
        --seed 0 \
        --num_inference_steps 4 \
        2>&1 | tee -a "$LOG"

    # ---- Eval ----
    $PYTHON eval.py \
        --gen "$GEN_OUT" \
        --gt "$VAL_GT" \
        --output "$RESULT" \
        2>&1 | tee -a "$LOG"

    echo "[$(date)] $NAME — done"
done

# -----------------------------------------------------------------------------
# Aggregate leaderboard
# -----------------------------------------------------------------------------
echo "============================================================"
echo "HP SWEEP LEADERBOARD (sorted by PSNR descending)"
echo "============================================================"
$PYTHON - <<PYEOF
import json, os, glob
results_dir = "$OUT_ROOT/results"
rows = []
for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    d = json.load(open(f))
    rows.append({
        "config": os.path.basename(f).replace(".json", ""),
        "psnr": d["averages"]["psnr"],
        "ssim": d["averages"]["ssim"],
        "fid": d.get("fid", float("nan")),
    })
rows.sort(key=lambda r: -r["psnr"])
print(f'{"Config":<18} {"PSNR":>8} {"SSIM":>8} {"FID":>8}')
print("-" * 46)
for r in rows:
    print(f'{r["config"]:<18} {r["psnr"]:>8.4f} {r["ssim"]:>8.4f} {r["fid"]:>8.2f}')

# Save aggregated CSV
with open(os.path.join("$OUT_ROOT", "leaderboard.csv"), "w") as f:
    f.write("config,psnr,ssim,fid\n")
    for r in rows:
        f.write(f'{r["config"]},{r["psnr"]:.4f},{r["ssim"]:.4f},{r["fid"]:.2f}\n')
print(f'\nSaved CSV: $OUT_ROOT/leaderboard.csv')
PYEOF
