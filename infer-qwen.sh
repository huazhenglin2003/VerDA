#!/usr/bin/env bash
set -euo pipefail

# ============================== Paths ==============================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# TODO: Set MODEL_CACHE_ROOT to the directory containing the downloaded model
# checkpoints if they are not stored under ./models.
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$ROOT_DIR/models}"
# TODO: Set OUTPUT_ROOT if inference results should be written outside ./outputs.
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
export VERDA_DATA_ROOT="${VERDA_DATA_ROOT:-$ROOT_DIR/data}"

POPE_QUESTION_ROOT="$VERDA_DATA_ROOT/pope/coco"
COCO_ANNOTATION_ROOT="$VERDA_DATA_ROOT/coco/annotations"

CONFIG_PATH="$ROOT_DIR/configs/verda_configs.yaml"
EVAL_MODULE="eval.eval"

# ============================== Utils ==============================
resolve_main_py() {
    local model_name="${1,,}"

    if [[ "$model_name" == llava-1.5-* || "$model_name" == *llava-next* ]]; then
        echo "$ROOT_DIR/scripts/llava_series/llava_main.py"
    elif [[ "$model_name" == qwen2.5-vl-* ]]; then
        echo "$ROOT_DIR/scripts/qwenvl_series/qwen2_5vl_main.py"
    elif [[ "$model_name" == qwen3-vl-* ]]; then
        echo "$ROOT_DIR/scripts/qwenvl_series/qwen3vl_main.py"
    elif [[ "$model_name" == internvl* ]]; then
        echo "$ROOT_DIR/scripts/internvl_series/internvl_main.py"
    elif [[ "$model_name" == instructblip-* ]]; then
        echo "$ROOT_DIR/scripts/instructblip/instructblip_main.py"
    elif [[ "$model_name" == glm-4v-* ]]; then
        echo "$ROOT_DIR/scripts/glm_series/glm_main.py"
    else
        echo "[ERROR] Unsupported model: $1" >&2
        return 1
    fi
}

# ========================== Inference + Eval ==========================
run_benchmark() {
    local model_dir=$1
    local dataset=$2
    local seed=$3
    local subset="${4:-}"

    local model_name="${model_dir##*--}"
    local model_path="$MODEL_CACHE_ROOT/$model_dir"

    local main_py
    main_py="$(resolve_main_py "$model_name")"

    local output_dir
    local output_file
    local full_output_path

    local -a inference_extra_args=()
    local -a method_args=()

    # ---------------------- Dataset config ----------------------
    if [[ "$dataset" == "chair" ]]; then

        output_dir="$OUTPUT_ROOT/$model_name/chair/$method"
        output_file="${dataset}-${method}-${max_new_tokens}-${seed}.json"

    elif [[ "$dataset" == "pope" ]]; then

        if [[ -z "$subset" ]]; then
            echo "[ERROR] POPE requires a subset: random | adversarial | popular" >&2
            return 1
        fi

        output_dir="$OUTPUT_ROOT/$model_name/pope/$method"
        output_file="${subset}-${dataset}-${method}-${max_new_tokens}-${seed}.json"

        inference_extra_args=(
            --subset "$subset"
        )

    else
        echo "[ERROR] Unsupported dataset: $dataset (expected: chair or pope)" >&2
        return 1
    fi

    # ----------------------- Method config ----------------------
    if [[ "$method" == "greedy" ]]; then

        method_args=()

    elif [[ "$method" == "verda" ]]; then

        method_args=(
            --use_verda
            --config_path "$CONFIG_PATH"
        )

    else
        echo "[ERROR] Unsupported method: $method (expected: greedy or verda)" >&2
        return 1
    fi

    full_output_path="$output_dir/$output_file"

    echo "============================================================"
    echo "Model   : $model_name"
    echo "Dataset : $dataset${subset:+/$subset}"
    echo "Method  : $method"
    echo "Main    : $main_py"
    echo "Output  : $full_output_path"
    echo "============================================================"

    # ------------------------- Inference -------------------------
    if [[ ! -f "$full_output_path" ]]; then
        mkdir -p "$output_dir"

        if (( num_chunks > ${#GPUS[@]} )); then
            echo "[ERROR] num_chunks=$num_chunks exceeds available GPUS=${#GPUS[@]}" >&2
            return 1
        fi

        local -a pids=()
        local i gpu_id

        for ((i = 0; i < num_chunks; i++)); do
            gpu_id="${GPUS[$i]}"

            python "$main_py" \
                --model_path "$model_path" \
                --dataset "$dataset" \
                --output_file "$output_dir/$i-$output_file" \
                --num_chunks "$num_chunks" \
                --max_new_tokens "$max_new_tokens" \
                --seed "$seed" \
                --device "cuda:$gpu_id" \
                --chunk_idx "$i" \
                "${method_args[@]}" \
                "${inference_extra_args[@]}" &

            pids+=("$!")
        done

        local failed=0
        local pid

        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done

        if (( failed != 0 )); then
            echo "[ERROR] At least one inference process failed." >&2
            return 1
        fi

        : > "$full_output_path"

        for ((i = 0; i < num_chunks; i++)); do
            cat "$output_dir/$i-$output_file" >> "$full_output_path"
            rm "$output_dir/$i-$output_file"
        done

    else
        echo "[SKIP] Inference result already exists."
    fi

    # ---------------------------- Eval ---------------------------
    if [[ "$dataset" == "chair" ]]; then

        python -m "$EVAL_MODULE" \
            --eval_chair \
            --result-file "$full_output_path" \
            --anno-file "$COCO_ANNOTATION_ROOT/captions_val2014.json" \
            --anno-dir "$COCO_ANNOTATION_ROOT"

        echo "[DONE] CHAIR result: $full_output_path"

    elif [[ "$dataset" == "pope" ]]; then

        local question_file="$POPE_QUESTION_ROOT/coco_pope_${subset}.json"
        local metric_file="$output_dir/${subset}-metric.json"

        python -m "$EVAL_MODULE" \
            --question-file "$question_file" \
            --result-file "$full_output_path" \
            --eval_pope \
            --pope_savepath "$metric_file"

        echo "[DONE] POPE metric: $metric_file"
    fi
}

# =============================== Run ===============================
# TODO: Keep only the checkpoint directory names available under
# MODEL_CACHE_ROOT, or replace these entries with your local directory names.
MODEL_DIRS=(
    "models--Qwen--Qwen2.5-VL-7B-Instruct"
    "models--Qwen--Qwen3-VL-8B-Instruct"
)

METHODS=("greedy" "verda")
dataset="pope"  # pope | chair
max_new_tokens=512
seed=0
# TODO: Update the GPU IDs for the local machine. num_chunks follows this list.
GPUS=(0 1 2 3 4 5 6 7)
num_chunks="${#GPUS[@]}"

POPE_SUBSETS=(random adversarial popular)

for model_dir in "${MODEL_DIRS[@]}"; do
    for method in "${METHODS[@]}"; do
        # ===================== POPE ====================
        for subset in "${POPE_SUBSETS[@]}"; do
            run_benchmark \
                "$model_dir" \
                "$dataset" \
                "$seed" \
                "$subset"
        done
    done
done
