#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29689}"
DATASET_DIRS="${DATASET_DIRS:?Set DATASET_DIRS to the four comma-separated LIBERO RLDS directories}"
DINOV3_PATH="${DINOV3_PATH:-$PROJECT_ROOT/pretrained/dinov3-vitb16}"
BERT_PATH="${BERT_PATH:-$PROJECT_ROOT/pretrained/bert-base-uncased}"
R3M_PATH="${R3M_PATH:-$PROJECT_ROOT/pretrained/r3m-resnet18/backbone.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/outputs/finalvla_from0_100k}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-finalvla_step}"
RESUME_MODE="${RESUME_MODE:-none}"
R3M_FEATURE_CACHE_PATH="${R3M_FEATURE_CACHE_PATH:-}"

export LIBRARY_PATH="/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
CACHE_ARGS=()
if [[ -n "$R3M_FEATURE_CACHE_PATH" ]]; then
  CACHE_ARGS=(--r3m_feature_cache_path "$R3M_FEATURE_CACHE_PATH")
fi

IFS=',' read -r -a GPU_ARRAY <<< "$TRAIN_GPUS"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"

if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "TRAIN_GPUS must contain at least one GPU" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$R3M_PATH" ]]; then
  echo "R3M checkpoint not found: $R3M_PATH" >&2
  exit 2
fi
if [[ ! -f "$DINOV3_PATH/model.safetensors" ]]; then
  echo "DINOv3 checkpoint not found: $DINOV3_PATH/model.safetensors" >&2
  exit 2
fi
if [[ ! -f "$BERT_PATH/model.safetensors" ]]; then
  echo "BERT checkpoint not found: $BERT_PATH/model.safetensors" >&2
  exit 2
fi
if [[ "$RESUME_MODE" != "none" && "$RESUME_MODE" != "model" && "$RESUME_MODE" != "all" ]]; then
  echo "RESUME_MODE must be one of: none, model, all" >&2
  exit 2
fi
if [[ "$RESUME_MODE" == "none" ]] && compgen -G "$CHECKPOINT_DIR/${CHECKPOINT_PREFIX}_*.pth" >/dev/null; then
  echo "Checkpoint directory is not empty: $CHECKPOINT_DIR" >&2
  exit 2
fi
if [[ "$RESUME_MODE" != "none" ]] && ! compgen -G "$CHECKPOINT_DIR/${CHECKPOINT_PREFIX}_*.pth" >/dev/null; then
  echo "No checkpoint found to resume in: $CHECKPOINT_DIR" >&2
  exit 2
fi

mkdir -p "$CHECKPOINT_DIR"

exec env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  experiments/libero/train_history12_dinov3_r3m_controlled_causal_state_v15.py \
  --dataset_dirs "$DATASET_DIRS" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path "$DINOV3_PATH" \
  --bert_path "$BERT_PATH" \
  --r3m_path "$R3M_PATH" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --checkpoint_prefix "$CHECKPOINT_PREFIX" \
  --resume_mode "$RESUME_MODE" \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --lr 5e-5 \
  --weight_decay 1e-10 \
  --head_lr 5e-5 \
  --dinov3_lr 5e-5 \
  --head_weight_decay 1e-10 \
  --dinov3_weight_decay 1e-10 \
  --max_steps 100000 \
  --lr_schedule_steps 100000 \
  --warmup_steps 12500 \
  --min_lr_ratio 0.0 \
  --save_steps 10000 \
  --log_freq 20 \
  --max_grad_norm 1.0 \
  --num_workers 4 \
  --shuffle_buffer 512 \
  --step_mix_buffer_size 64 \
  --seed 42 \
  "${CACHE_ARGS[@]}" \
  "$@"
