#!/bin/bash
#
# IntRR Training Script
# Based on the GRID framework by Snap Research
# Train IntRR models with different SID types
#
# Usage: ./run_intrr.sh --datasets DATASETS --seeds SEEDS --sid-type TYPES [OPTIONS]
#
# Examples:
#   ./run_intrr.sh --datasets toys --seeds 42 --sid-type rkmeans
#   ./run_intrr.sh --datasets toys,beauty,sports --sid-type rkmeans,rqvae
#

set -euo pipefail

# Load dataset configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/configs/dataset_config.sh"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Defaults
readonly DEFAULT_DATASETS="toys"
readonly DEFAULT_SEEDS="42"
readonly DEFAULT_MAX_PARALLEL=4

# Logging functions
log_info() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"; }
log_error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"; }

print_usage() {
    cat << EOF
Usage: $0 --sid-type TYPES [OPTIONS]

Required:
    --sid-type TYPES        SID types (comma-separated): rkmeans, rqvae, vqvae

Options:
    -h, --help              Show this help message
    --datasets DATASETS     Datasets (comma-separated, default: $DEFAULT_DATASETS)
    --seeds SEEDS           Random seeds (comma-separated, default: $DEFAULT_SEEDS)
    --max-parallel NUM      Max parallel jobs (default: $DEFAULT_MAX_PARALLEL)

Examples:
    $0 --datasets toys --seeds 42 --sid-type rkmeans
    $0 --datasets toys,beauty,sports --sid-type rkmeans,rqvae
EOF
    exit 1
}

# Parse arguments

DATASETS="$DEFAULT_DATASETS"
SEEDS="$DEFAULT_SEEDS"
SID_TYPES=""
MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) print_usage ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --sid-type) SID_TYPES="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        *) log_error "Unknown option: $1"; print_usage ;;
    esac
done

[[ -z "$SID_TYPES" ]] && { log_error "SID type required (--sid-type)"; print_usage; }

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=',' read -ra SID_TYPE_ARRAY <<< "$SID_TYPES"

# Validate inputs
for sid_type in "${SID_TYPE_ARRAY[@]}"; do
    [[ ! "$sid_type" =~ ^(rkmeans|rqvae|vqvae)$ ]] && { log_error "Invalid SID type: $sid_type"; exit 1; }
done

for dataset in "${DATASET_ARRAY[@]}"; do
    [[ ! -d "data/amazon_data/$dataset" ]] && { log_error "Data not found: data/amazon_data/$dataset"; exit 1; }
    for sid_type in "${SID_TYPE_ARRAY[@]}"; do
        get_dataset_config "$dataset" "$sid_type" > /dev/null || { log_error "Unsupported: $dataset/$sid_type"; exit 1; }
    done
done

# Setup experiment

LOG_DIR="logs/intrr_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

log_info "=========================================="
log_info "IntRR Training"
log_info "=========================================="
log_info "Datasets:     ${DATASET_ARRAY[*]}"
log_info "Seeds:        ${SEED_ARRAY[*]}"
log_info "SID Types:    ${SID_TYPE_ARRAY[*]}"
log_info "Max Parallel: $MAX_PARALLEL"
log_info "Log Dir:      $LOG_DIR"
log_info "=========================================="

# Generate experiment list
declare -a EXPERIMENTS
for ds in "${DATASET_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        for sid_type in "${SID_TYPE_ARRAY[@]}"; do
            EXPERIMENTS+=("${ds}|${seed}|${sid_type}")
        done
    done
done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
log_info "Total experiments: $TOTAL_EXPERIMENTS"
log_info "=========================================="

RUNNING_PIDS=()
RUNNING_NAMES=()
FAILED_TASKS=0
COMPLETED_TASKS=0

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; do
        for i in "${!RUNNING_PIDS[@]}"; do
            if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
                wait "${RUNNING_PIDS[$i]}"
                if [[ $? -eq 0 ]]; then
                    log_info "  ✓ ${RUNNING_NAMES[$i]} completed"
                else
                    log_error "  ✗ ${RUNNING_NAMES[$i]} failed"
                    FAILED_TASKS=$((FAILED_TASKS + 1))
                fi
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
                unset 'RUNNING_PIDS[i]' 'RUNNING_NAMES[i]'
                RUNNING_PIDS=("${RUNNING_PIDS[@]}")
                RUNNING_NAMES=("${RUNNING_NAMES[@]}")
                return
            fi
        done
        sleep 5
    done
}

start_experiment() {
    local dataset=$1 seed=$2 sid_type=$3
    local config exp_name log_file
    
    config=$(get_dataset_config "$dataset" "$sid_type")
    IFS='|' read -r total_item_num embedding_path sid_path <<< "$config"
    
    exp_name="${dataset}_seed${seed}_${sid_type}"
    log_file="$LOG_DIR/${exp_name}.log"
    
    log_info "Starting: $exp_name"
    
    python -m src.train experiment=intrr_train_flat \
        data_dir="$dataset" \
        sid_type="$sid_type" \
        semantic_id_path="$sid_path" \
        embedding_path="$embedding_path" \
        total_item_id="$total_item_num" \
        seed="$seed" \
        ++should_skip_retry=True > "$log_file" 2>&1 &
    
    RUNNING_PIDS+=("$!")
    RUNNING_NAMES+=("$exp_name")
    log_info "  PID: $!, Log: $log_file"
}

# Main execution

log_info "Starting experiments..."

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r dataset seed sid_type <<< "$exp"
    wait_for_slot
    start_experiment "$dataset" "$seed" "$sid_type"
    log_info "Progress: $((COMPLETED_TASKS + ${#RUNNING_PIDS[@]}))/$TOTAL_EXPERIMENTS"
done

log_info "All experiments started, waiting for completion..."
log_info "Monitor logs: tail -f $LOG_DIR/*.log"

for i in "${!RUNNING_PIDS[@]}"; do
    wait "${RUNNING_PIDS[$i]}"
    if [[ $? -eq 0 ]]; then
        log_info "  ✓ ${RUNNING_NAMES[$i]} completed"
    else
        log_error "  ✗ ${RUNNING_NAMES[$i]} failed"
        FAILED_TASKS=$((FAILED_TASKS + 1))
    fi
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
done

# Summary
log_info "=========================================="
log_info "Training Complete"
log_info "=========================================="
log_info "Total:   $TOTAL_EXPERIMENTS"
log_info "Success: $((TOTAL_EXPERIMENTS - FAILED_TASKS))"
log_info "Failed:  $FAILED_TASKS"
log_info "Logs:    $LOG_DIR"
log_info "=========================================="

[[ $FAILED_TASKS -gt 0 ]] && { log_error "$FAILED_TASKS experiments failed. Check logs in: $LOG_DIR/"; exit 1; }

log_info "All experiments completed successfully!"
exit 0
