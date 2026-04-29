# Grayscale → IR with FLUX.2-Klein-4B

Train a FLUX.2-Klein-4B DreamBooth LoRA to convert grayscale images of maritime vessels into infrared (IR) images. The grayscale conditioning data is derived from real IR by passing it through FLUX.2-Klein-9B and converting the output to grayscale.

## Pipeline

```
real IR  →[Klein-9B → grayscale]→  synthetic_grayscale  →[Klein-4B + LoRA]→  generated IR
```

`run_flux2.py` always saves a single-channel grayscale image regardless of the FLUX model used. The LoRA learns the mapping `grayscale → IR`.

## Setup

```bash
conda create -n flux-dsta python=3.12 -y
conda activate flux-dsta

# Install matched torch/torchvision (cu124 example)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Other deps
pip install diffusers transformers accelerate peft prodigyopt safetensors datasets pillow scikit-image torchmetrics torch-fidelity tabulate tqdm pandas

# Login to HF (Klein models are gated)
huggingface-cli login

# Git LFS — required to clone the LoRA weights in models/
git lfs install
```

Set HF cache:
```bash
export HF_HOME=/path/to/your/hf_cache
```

## Step 1: Synthesize Grayscale from Real IR (Klein-9B)

Use Klein-9B with category-specific prompts. Output is grayscale (single channel). Run for both `train` and `test` splits.

```bash
# Run for SPLIT in {train, test}
python run_flux2.py \
  --input_folder  /path/to/real_ir/<SPLIT>/<category> \
  --output_folder /path/to/synthetic_grayscale/<SPLIT>/<category> \
  --model_name "black-forest-labs/FLUX.2-klein-9B" \
  --prompt "Turn this thermal infrared image of a <vessel_type> into a visually realistic RGB image as it would appear under visible light. Accurately preserve the vessel's hull structure, silhouette, and outline." \
  --num_inference_steps 20 \
  --guidance_scale 1.0 \
  --seed 0
```

Run per-category for best quality (vessel-specific prompts). Produces `synthetic_grayscale/train/` and `synthetic_grayscale/test/`.

## Step 2: Build HF Dataset (synthetic grayscale → real IR, train split)

```bash
python create_hf_dataset.py \
  --cond_dir /path/to/synthetic_grayscale/train \
  --target_dir /path/to/real_ir/train \
  --output_dir /path/to/HFDataset_rgb2ir \
  --prompt "turn the visible image of Marine Vessel into sks infrared"
```

`cond_dir` and `target_dir` must mirror each other (same filenames, same category subdirs).

## Step 3: Train LoRA (best config: Prodigy + rank 64 + 150 epochs + dropout)

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/path/to/your/hf_cache

accelerate launch \
  --num_processes 1 \
  --main_process_port 29500 \
  train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --dataset_name /path/to/HFDataset_rgb2ir \
  --image_column target_image \
  --cond_image_column cond_image \
  --instance_prompt "turn the visible image of Marine Vessel into sks infrared" \
  --output_dir /path/to/flux2-klein4b-rgb2ir-lora \
  --resolution 256 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1.0 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --num_train_epochs 150 \
  --rank 64 --lora_alpha 64 \
  --lora_layers to_k,to_q,to_v,to_out.0,to_qkv_mlp_proj,add_q_proj,add_k_proj,add_v_proj,to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2 \
  --lora_dropout 0.05 \
  --guidance_scale 1.0 \
  --optimizer prodigy \
  --weighting_scheme cosmap \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --cache_latents \
  --seed 42 \
  --report_to none \
  --allow_tf32 \
  --random_flip
```

## Step 4: Inference (grayscale → IR)

```bash
python run_flux2.py \
  --lora_weights /path/to/flux2-klein4b-rgb2ir-lora \
  --input_folder /path/to/synthetic_grayscale/test \
  --output_folder /path/to/output_ir \
  --prompt "turn the visible image of Marine Vessel into sks infrared" \
  --model_name "black-forest-labs/FLUX.2-klein-4B" \
  --guidance_scale 1.0 \
  --seed 0 \
  --num_inference_steps 4
```

## Step 5: Evaluate

```bash
python eval.py \
  --gen /path/to/output_ir \
  --gt /path/to/real_ir/test
```

Outputs PSNR, SSIM, FID. Saves per-image results to `eval_results.json`.

## Hyperparameter Tuning (optional)

After Step 2 produces the HF dataset, run a sweep over informed-prior HP variations:

```bash
# Edit DATASET / VAL_INPUT / VAL_GT / OUT_ROOT inside the script first
bash scripts/hp_tune.sh 0   # GPU 0
```

The sweep covers rank (32/64/128), epochs (100/150/200), dropout (0.05/0.10), LoRA targets (attention-only vs +FF), and weighting scheme (cosmap vs sigma_sqrt). Skips already-completed configs. Outputs `leaderboard.csv` ranked by PSNR.

## Pretrained LoRA

A pretrained grayscale → IR LoRA (PSNR=15.97, SSIM=0.521, FID=55.37) is included at `models/pytorch_lora_weights.safetensors` via Git LFS. To use it, clone the repo with `git lfs install` first (or run `git lfs pull` after a regular clone). Then point `--lora_weights /path/to/RGB2IR/models` in Step 4.

## File Reference

| File | Purpose |
|------|---------|
| `create_hf_dataset.py` | Build HuggingFace dataset from paired cond+target image directories |
| `run_flux2.py` | Inference (always saves grayscale output) |
| `train_dreambooth_lora_flux2_klein_img2img.py` | DreamBooth LoRA training script |
| `eval.py` | PSNR / SSIM / FID evaluation |
| `scripts/hp_tune.sh` | HP sweep script for the training step (sequential, idempotent) |
