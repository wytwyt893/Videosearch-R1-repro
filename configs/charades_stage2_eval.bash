#!/usr/bin/env bash
# Charades Stage2 evaluation protocol. Source scripts/env.sh before this file.
# Sourcing this file sets parameters only; it does not download or run a model.
if [[ -z "${VIDEOSEARCH_WORKSPACE:-}" ]]; then
  echo 'Source scripts/env.sh before configs/charades_stage2_eval.bash' >&2
  return 2 2>/dev/null || exit 2
fi

export VIDEOSEARCH_STAGE2_CHECKPOINT="${VIDEOSEARCH_WORKSPACE}/checkpoints/VideoSearchR1__charades-stage2"
export QUERY_EMBEDDER_MODEL_PATH="${VIDEOSEARCH_WORKSPACE}/checkpoints/Qwen__Qwen3-VL-Embedding-2B"

# Paper-explicit inference choices.
export MAX_TURN=2
export REFINE_TOKEN_COUNT=8
export TEMPORAL=True
export TOPK=1,5,10,100
export IOU_THRESHOLDS=0.3,0.5,0.7

# Full official test set. This deliberately overrides the smoke limit of five.
export MAX_SAMPLES=0
export EVAL_LIMIT_RATIO=1.0
export RERANKS=all
export RESUME_FROM_JSONL=False
export SFT_FORCE_DATASET_DEFAULTS=True
export DEFAULT_CHECKPOINT=charades-stage2

# Local scoring policy chosen by the user. The official evaluator does not
# consume these flags; the local scorer and configuration checker enforce them.
export VIDEOSEARCH_VER_TURN=last_verified
export VIDEOSEARCH_VER_DENOMINATOR=valid_answers
export VIDEOSEARCH_TERMINAL_REFINEMENT=keep_official

# Paper-unspecified implementation choices: retain the official shell entry.
export USE_UPDATED_QUERY=False
export USE_QUERY_EMBEDDER_PATH=True
export USE_REFINE_GATE=False
export REFINE_TOKEN='<REFINE>'
export QFINAL_POOLING=latent_last
export QFINAL_NORMALIZE=True
export QUERY_EMBEDDER_MAX_LENGTH=128
export QUERY_EMBEDDER_INPUT_PREFIX="Represent the user's input"
export MODEL_MAX_LENGTH=20384
export EXTERNAL_EVAL_MAX_NEW_TOKENS=1024
export BF16=True
export USE_VLLM=True
export SEED=42
# The official LLM constructor does not pass SEED to vLLM; the smoke engine
# logged seed=0. Do not describe this as every component using seed=42.
export VLLM_MODEL_PATH=''
export VLLM_TENSOR_PARALLEL_SIZE=1
export NUM_PROCESSES_PER_GPU=1
export VLLM_MAX_NUM_SEQS=1
export VLLM_EVAL_BATCH_SIZE=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=False

# Local execution choices retained from the successful smoke, not paper values.
export VLLM_GPU_MEMORY_UTILIZATION=0.45
export VLLM_ENFORCE_EAGER=True
export SAVE_QUERY_VECTORS=False
export PROFILE_ROLLOUT_DEPTHS=''

# Official visual budgets and terminal SQR behavior remain unchanged. Score
# saved predictions with scripts/score_charades_protocol.py; selecting final
# VER must not pair an old video's answer with a newly retrieved video's label.
