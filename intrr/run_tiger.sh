#!/bin/bash
#
# ===========================
# Usage:
#   ./run_tiger.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   --datasets DATASETS     Comma-separated list of datasets (default: toys)
#   --seeds SEEDS           Comma-separated list of seeds (default: 42)
#   --sid-type TYPES        Comma-separated list of SID types (REQUIRED)
#                           Available: rkmeans, rqvae, vqvae
#   --max-parallel NUM      Maximum parallel experiments (default: 3)
#
# Examples:
#   ./run_tiger.sh --datasets toys --seeds 42 --sid-type rkmeans
#   ./run_tiger.sh --datasets toys,beauty,sports --sid-type rkmeans,rqvae
#

# ==============================================================================

# Load dataset configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/configs/dataset_config.sh"

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

DEFAULT_DATASETS="toys"
DEFAULT_SEEDS="42"
DEFAULT_MAX_PARALLEL=3

# ==============================================================================
# Utility Functions
# ==============================================================================

log_info() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"; }
log_error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"; }

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    --datasets DATASETS     Comma-separated list of datasets (default: $DEFAULT_DATASETS)
    --seeds SEEDS           Comma-separated list of seeds (default: $DEFAULT_SEEDS)
    --sid-type TYPES        Comma-separated list of SID types (REQUIRED)
                            Available: rkmeans, rqvae, vqvae
    --max-parallel NUM      Maximum parallel experiments (default: $DEFAULT_MAX_PARALLEL)

Supported datasets: toys, beauty, sports
Supported SID types: rkmeans, rqvae, vqvae

Examples:
    $0 --datasets toys --seeds 42 --sid-type rkmeans
    $0 --datasets toys,beauty,sports --sid-type rkmeans,rqvae
EOF
    exit 1
}

# ==============================================================================
# Argument Parsing
# ==============================================================================

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

# Validate SID types
for sid_type in "${SID_TYPE_ARRAY[@]}"; do
    case $sid_type in
        "rkmeans"|"rqvae"|"vqvae") ;;
        *) log_error "Invalid SID type: $sid_type"; exit 1 ;;
    esac
done

# Validate datasets and SID files
for dataset in "${DATASET_ARRAY[@]}"; do
    [[ ! -d "data/amazon_data/$dataset" ]] && { log_error "Data not found: data/amazon_data/$dataset"; exit 1; }
    for sid_type in "${SID_TYPE_ARRAY[@]}"; do
        config=$(get_dataset_config "$dataset" "$sid_type")
        [[ -z "$config" ]] && { log_error "Unsupported dataset: $dataset"; exit 1; }
        IFS='|' read -r _ embedding_path sid_path <<< "$config"
        [[ ! -f "$sid_path" ]] && { log_error "SID file not found: $sid_path"; exit 1; }
    done
done

# ==============================================================================
# Experiment Management
# ==============================================================================

LOG_DIR="logs/tiger_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

log_info "=========================================="
log_info "TIGER Baseline Training"
log_info "=========================================="
log_info "Datasets:     ${DATASET_ARRAY[*]}"
log_info "Seeds:        ${SEED_ARRAY[*]}"
log_info "SID Types:    ${SID_TYPE_ARRAY[*]}"
log_info "Max Parallel: $MAX_PARALLEL"
log_info "Log Dir:      $LOG_DIR"
log_info "=========================================="

# Generate experiment combinations
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

declare -a RUNNING_PIDS RUNNING_NAMES
FAILED_TASKS=0
COMPLETED_TASKS=0

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; do
        for i in "${!RUNNING_PIDS[@]}"; do
            if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
                wait "${RUNNING_PIDS[$i]}"
                local ec=$?
                [[ $ec -eq 0 ]] && log_info "  Done: ${RUNNING_NAMES[$i]}" || { log_error "  Failed: ${RUNNING_NAMES[$i]} (exit $ec)"; FAILED_TASKS=$((FAILED_TASKS + 1)); }
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
                unset 'RUNNING_PIDS[i]' 'RUNNING_NAMES[i]'
                RUNNING_PIDS=("${RUNNING_PIDS[@]}"); RUNNING_NAMES=("${RUNNING_NAMES[@]}")
                return
            fi
        done
        sleep 5
    done
}

start_experiment() {
    local dataset=$1 seed=$2 sid_type=$3
    local config=$(get_dataset_config "$dataset" "$sid_type")
    IFS='|' read -r _ embedding_path sid_path <<< "$config"
    
    local exp_name="${dataset}_seed${seed}_${sid_type}"
    local log_file="$LOG_DIR/${exp_name}.log"
    
    log_info "Starting: $exp_name"
    
    # Only pass dataset-specific parameters - all others are fixed in config
    python -m src.train "experiment=decoder_train_flat" \
        "data_dir=$dataset" \
        "sid_type=$sid_type" \
        "semantic_id_path=$sid_path" \
        "seed=$seed" \
        "++should_skip_retry=True" > "$log_file" 2>&1 &
    
    RUNNING_PIDS+=("$!")
    RUNNING_NAMES+=("$exp_name")
    log_info "  PID: $!, Log: $log_file"
}

# ==============================================================================
# Main Execution
# ==============================================================================

log_info "Starting experiments..."

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r dataset seed sid_type <<< "$exp"
    wait_for_slot
    start_experiment "$dataset" "$seed" "$sid_type"
    log_info "Progress: $((COMPLETED_TASKS + ${#RUNNING_PIDS[@]}))/$TOTAL_EXPERIMENTS"
done

log_info "All started, waiting for completion..."
log_info "Monitor: tail -f $LOG_DIR/*.log"

for i in "${!RUNNING_PIDS[@]}"; do
    wait "${RUNNING_PIDS[$i]}"
    local ec=$?
    [[ $ec -eq 0 ]] && log_info "  Done: ${RUNNING_NAMES[$i]}" || { log_error "  Failed: ${RUNNING_NAMES[$i]} (exit $ec)"; FAILED_TASKS=$((FAILED_TASKS + 1)); }
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
done

# ==============================================================================
# Summary
# ==============================================================================

log_info "=========================================="
log_info "Training Complete"
log_info "=========================================="
log_info "Total:   $TOTAL_EXPERIMENTS"
log_info "Success: $((TOTAL_EXPERIMENTS - FAILED_TASKS))"
log_info "Failed:  $FAILED_TASKS"
log_info "Logs:    $LOG_DIR"
log_info "=========================================="

[[ $FAILED_TASKS -gt 0 ]] && log_error "$FAILED_TASKS experiments failed. Check: $LOG_DIR/"

exit $FAILED_TASKS
