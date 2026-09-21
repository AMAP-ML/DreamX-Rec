#!/bin/bash
#
# This script runs the complete pipeline for generating Semantic IDs:
#   1. Generate embeddings from pre-trained language models
#   2. Train Semantic ID models (RKMeans/RQVAE/RVQ)
#   3. Generate Semantic IDs for all items
#
# Usage:
#   ./gen_sid.sh [OPTIONS]
#
# Options:
#   -h, --help                Show this help message
#   --datasets DATASETS       Comma-separated list of datasets (default: sports,beauty,toys)
#   --sid-methods METHODS     Comma-separated list of SID methods (default: rkmeans,rqvae,rvq)
#   --embedding-model MODEL   Embedding model name (default: google/flan-t5-xl)
#   --embedding-dim DIM       Embedding dimension (default: 2048)
#   --hierarchies NUM         Number of hierarchies (default: 3)
#   --codebook-width WIDTH    Codebook width (default: 128)
#
# Examples:
#   ./gen_sid.sh --datasets sports,beauty --sid-methods rkmeans,rqvae
#   ./gen_sid.sh --datasets toys --sid-methods rvq --hierarchies 4
#

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'  # No Color

# Default parameters
DEFAULT_DATASETS="sports,beauty,toys"
DEFAULT_SID_METHODS="rkmeans,rqvae,rvq"
DEFAULT_EMBEDDING_MODEL="google/flan-t5-xl"
DEFAULT_EMBEDDING_DIM=2048
DEFAULT_NUM_HIERARCHIES=3
DEFAULT_CODEBOOK_WIDTH=128

# ==============================================================================
# Utility Functions
# ==============================================================================

log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help                Show this help message
    --datasets DATASETS       Comma-separated list of datasets (default: $DEFAULT_DATASETS)
    --sid-methods METHODS     Comma-separated list of SID methods (default: $DEFAULT_SID_METHODS)
    --embedding-model MODEL   Embedding model name (default: $DEFAULT_EMBEDDING_MODEL)
    --embedding-dim DIM       Embedding dimension (default: $DEFAULT_EMBEDDING_DIM)
    --hierarchies NUM         Number of hierarchies (default: $DEFAULT_NUM_HIERARCHIES)
    --codebook-width WIDTH    Codebook width (default: $DEFAULT_CODEBOOK_WIDTH)

Supported datasets: sports, beauty, toys
Supported SID methods: rkmeans, rqvae, rvq

Examples:
    $0 --datasets sports,beauty --sid-methods rkmeans,rqvae
    $0 --datasets toys --sid-methods rvq --hierarchies 4
EOF
    exit 1
}

# Get item count for a dataset
get_item_count() {
    local dataset=$1
    case $dataset in
        "sports") echo 18357 ;;
        "toys")   echo 11924 ;;
        "beauty") echo 12101 ;;
        *)
            log_error "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
}

# Get the most recent log directory
get_latest_log_dir() {
    ls -td logs/*/*/output/* 2>/dev/null | head -n1 | xargs
}

# ==============================================================================



# ==============================================================================
# Argument Parsing
# ==============================================================================

DATASETS="$DEFAULT_DATASETS"
SID_METHODS="$DEFAULT_SID_METHODS"
EMBEDDING_MODEL="$DEFAULT_EMBEDDING_MODEL"
EMBEDDING_DIM="$DEFAULT_EMBEDDING_DIM"
NUM_HIERARCHIES="$DEFAULT_NUM_HIERARCHIES"
CODEBOOK_WIDTH="$DEFAULT_CODEBOOK_WIDTH"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --sid-methods)
            SID_METHODS="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --embedding-dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --hierarchies)
            NUM_HIERARCHIES="$2"
            shift 2
            ;;
        --codebook-width)
            CODEBOOK_WIDTH="$2"
            shift 2
            ;;
        *)
            log_error "Unknown argument: $1"
            print_usage
            ;;
    esac
done

# Convert comma-separated strings to arrays
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra SID_METHOD_ARRAY <<< "$SID_METHODS"

# ==============================================================================
# Pipeline Functions
# ==============================================================================

# Run the complete pipeline for a single dataset and SID method
run_pipeline() {
    local dataset=$1
    local sid_method=$2
    local task_id=$3

    log_debug "----------------------------------------"
    log_debug "Task [$task_id/$TOTAL_TASKS]: dataset=$dataset, method=$sid_method"
    log_debug "----------------------------------------"

    # Validate data directory
    if [[ ! -d "data/amazon_data/$dataset" ]]; then
        log_error "Data directory for '$dataset' not found, skipping..."
        FAILED_TASKS+=("$dataset-$sid_method: data directory not found")
        return 1
    fi

    local total_item_num
    total_item_num=$(get_item_count "$dataset")

    # -------------------------------------------------------------------------
    # Step 1: Generate Embeddings
    # -------------------------------------------------------------------------
    log_info "Step 1/3: Generating embeddings..."
    local cmd="python -m src.inference experiment=sem_embeds_inference_flat data_dir=$dataset embedding_model=$EMBEDDING_MODEL"
    log_info "Command: $cmd"
    
    if ! $cmd; then
        log_error "Step 1 failed: embedding generation"
        FAILED_TASKS+=("$dataset-$sid_method: step 1 failed")
        return 1
    fi
    log_info "Step 1 completed ✓"

    # Get embedding output path
    local latest_log_dir
    latest_log_dir=$(get_latest_log_dir)
    if [[ -z "$latest_log_dir" ]]; then
        log_error "Cannot find log directory"
        FAILED_TASKS+=("$dataset-$sid_method: log directory not found")
        return 1
    fi

    local embedding_output_path="${latest_log_dir}/pickle/merged_predictions_tensor.pt"
    if [[ ! -f "$embedding_output_path" ]]; then
        log_error "Embedding output file not found: $embedding_output_path"
        FAILED_TASKS+=("$dataset-$sid_method: embedding output not found")
        return 1
    fi
    log_info "Embedding output: $embedding_output_path"

    # -------------------------------------------------------------------------
    # Step 2: Train Semantic ID Model
    # -------------------------------------------------------------------------
    log_info "Step 2/3: Training Semantic ID model ($sid_method)..."
    cmd="python -m src.train \
        experiment=${sid_method}_train_flat \
        data_dir=$dataset \
        embedding_path=$embedding_output_path \
        embedding_dim=$EMBEDDING_DIM \
        num_hierarchies=$NUM_HIERARCHIES \
        codebook_width=$CODEBOOK_WIDTH"
    log_info "Command: $cmd"
    
    if ! $cmd; then
        log_error "Step 2 failed: SID training"
        FAILED_TASKS+=("$dataset-$sid_method: step 2 failed")
        return 1
    fi
    log_info "Step 2 completed ✓"

    # Get checkpoint path
    latest_log_dir=$(get_latest_log_dir)
    local ckpt_dir="${latest_log_dir}/checkpoints/"
    local ckpt_path
    ckpt_path=$(ls -t "${ckpt_dir}"*.ckpt 2>/dev/null | head -n1)
    
    if [[ -z "$ckpt_path" ]]; then
        log_error "No checkpoint found in: $ckpt_dir"
        FAILED_TASKS+=("$dataset-$sid_method: checkpoint not found")
        return 1
    fi
    log_info "Checkpoint: $ckpt_path"

    # -------------------------------------------------------------------------
    # Step 3: Generate Semantic IDs
    # -------------------------------------------------------------------------
    log_info "Step 3/3: Generating Semantic IDs ($sid_method)..."
    cmd="python -m src.inference experiment=${sid_method}_inference_flat \
        data_dir=$dataset \
        embedding_path=$embedding_output_path \
        embedding_dim=$EMBEDDING_DIM \
        num_hierarchies=$NUM_HIERARCHIES \
        codebook_width=$CODEBOOK_WIDTH \
        ckpt_path=$ckpt_path"
    log_info "Command: $cmd"
    
    if ! $cmd; then
        log_error "Step 3 failed: SID generation"
        FAILED_TASKS+=("$dataset-$sid_method: step 3 failed")
        return 1
    fi
    log_info "Step 3 completed ✓"

    # Verify output
    latest_log_dir=$(get_latest_log_dir)
    local sid_output_path="${latest_log_dir}/pickle/merged_predictions_tensor.pt"
    if [[ ! -f "$sid_output_path" ]]; then
        log_error "SID output file not found: $sid_output_path"
        FAILED_TASKS+=("$dataset-$sid_method: SID output not found")
        return 1
    fi
    log_info "SID output: $sid_output_path"

    # Record success
    SUCCESS_TASKS+=("$dataset-$sid_method")
    log_info "Task completed ✓✓✓: $dataset-$sid_method"
    echo ""

    return 0
}

# ==============================================================================
# Main Execution
# ==============================================================================

# Create output directory
mkdir -p outputs

# Print execution plan
log_info "=========================================="
log_info "GRID Semantic ID Generation Pipeline"
log_info "=========================================="
log_info "Datasets:        ${DATASET_ARRAY[*]}"
log_info "SID Methods:     ${SID_METHOD_ARRAY[*]}"
log_info "Embedding Model: $EMBEDDING_MODEL"
log_info "Embedding Dim:   $EMBEDDING_DIM"
log_info "Hierarchies:     $NUM_HIERARCHIES"
log_info "Codebook Width:  $CODEBOOK_WIDTH"
log_info "=========================================="
echo ""

# Initialize task tracking
TOTAL_TASKS=$((${#DATASET_ARRAY[@]} * ${#SID_METHOD_ARRAY[@]}))
CURRENT_TASK=0
FAILED_TASKS=()
SUCCESS_TASKS=()

# Run pipeline for all combinations
for dataset in "${DATASET_ARRAY[@]}"; do
    for sid_method in "${SID_METHOD_ARRAY[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        run_pipeline "$dataset" "$sid_method" "$CURRENT_TASK"
    done
done

# ==============================================================================
# Summary
# ==============================================================================

echo ""
log_info "=========================================="
log_info "Pipeline Execution Complete"
log_info "=========================================="
log_info "Total tasks:     $TOTAL_TASKS"
log_info "Successful:      ${#SUCCESS_TASKS[@]}"
log_info "Failed:          ${#FAILED_TASKS[@]}"
echo ""

if [[ ${#SUCCESS_TASKS[@]} -gt 0 ]]; then
    log_info "Successful tasks:"
    for task in "${SUCCESS_TASKS[@]}"; do
        echo "  ✓ $task"
    done
    echo ""
fi

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    log_error "Failed tasks:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  ✗ $task"
    done
    echo ""
    exit 1
fi

log_info "All tasks completed successfully!"
log_info "=========================================="



