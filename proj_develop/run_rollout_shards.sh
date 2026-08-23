#!/usr/bin/env bash
# Fan out batch_infer.py across GPUs by sharding the dataset on row index.
# batch_infer already records category_id per row, so per-class analysis is done at
# analysis time by grouping the merged output -- no need to split the dataset by class.
#
# Run from the repo root. Each shard writes to
#   $OUTPUT_DIR_PREFIX/$EXPERIMENT_PREFIX_<start>_<end>/
# which is the same layout merge_dpo_dataset_chunks.py already expects.
set -euo pipefail

DATASET="${DATASET:-stabletext2brick_train.parquet}"
OUTPUT_DIR_PREFIX="${OUTPUT_DIR_PREFIX:-proj_develop/datasets/dpo_datasets}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-rollout}"
LOG_DIR="${LOG_DIR:-proj_develop/logs/rollout}"

# Row range of the dataset to cover: [START_IDX, START_IDX + TOTAL_ROWS).
# The full train parquet is 42604 rows; collecting all of it is very expensive
# (5 captions/row, each a full constrained generation), so default to a slice.
START_IDX="${START_IDX:-0}"
TOTAL_ROWS="${TOTAL_ROWS:-2000}"

# GPU indices to use, space separated.
GPUS=(${GPUS:-6 7 8 9})

mkdir -p "$LOG_DIR"

n_gpus=${#GPUS[@]}
# Ceiling division so the last shard picks up the remainder.
rows_per_shard=$(( (TOTAL_ROWS + n_gpus - 1) / n_gpus ))

echo "Dataset:       $DATASET"
echo "Rows:          [$START_IDX, $((START_IDX + TOTAL_ROWS)))"
echo "GPUs:          ${GPUS[*]}"
echo "Rows/shard:    $rows_per_shard"
echo

for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    shard_start=$(( START_IDX + i * rows_per_shard ))
    shard_end=$(( shard_start + rows_per_shard ))

    # Clamp the final shard to the requested range.
    if (( shard_start >= START_IDX + TOTAL_ROWS )); then
        continue
    fi
    if (( shard_end > START_IDX + TOTAL_ROWS )); then
        shard_end=$(( START_IDX + TOTAL_ROWS ))
    fi
    shard_rows=$(( shard_end - shard_start ))

    name="${EXPERIMENT_PREFIX}_$(printf '%06d' "$shard_start")_$(printf '%06d' "$shard_end")"
    echo "GPU $gpu -> rows [$shard_start, $shard_end)  ($shard_rows rows)  -> $name"

    # Both --start_idx and --max_rows must be passed: batch_infer ignores start_idx
    # entirely when max_rows is None.
    CUDA_VISIBLE_DEVICES="$gpu" nohup uv run batch_infer \
        --dataset "$DATASET" \
        --output_dir_prefix "$OUTPUT_DIR_PREFIX" \
        --experiment_name "$name" \
        --start_idx "$shard_start" \
        --max_rows "$shard_rows" \
        > "$LOG_DIR/${name}.log" 2>&1 &
done

echo
echo "Launched ${#GPUS[@]} shards. Logs: $LOG_DIR/"
echo "Wait for them with: wait"
wait
echo "All shards finished."
