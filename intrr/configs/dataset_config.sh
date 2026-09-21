#!/bin/bash
#
# Centralized dataset configuration for training scripts
#
# This file defines:
# - Item counts per dataset
# - LLM embedding paths
# - SID (Semantic ID) file paths
#

# ==============================================================================
# Dataset Item Counts
# ==============================================================================

declare -A DATASET_ITEM_COUNTS=(
    ["toys"]=11924
    ["beauty"]=12101
    ["sports"]=18357
)

# ==============================================================================
# LLM Embedding Paths
# ==============================================================================

declare -A DATASET_EMBEDDING_PATHS=(
    ["toys"]="logs/inference/runs/2025-12-03/17-39-24/pickle/merged_predictions_tensor.pt"
    ["beauty"]="logs/inference/runs/2026-01-21/11-47-30-beauty/pickle/merged_predictions_tensor.pt"
    ["sports"]="logs/inference/runs/2026-01-22/10-45-14-sports/pickle/merged_predictions_tensor.pt"
)

# ==============================================================================
# SID Path Pattern
# ==============================================================================

# Pattern: logs/inference/runs/output/{sid_type}-{dataset}-128/pickle/merged_predictions_tensor.pt
# Note: vqvae uses 'rvq' as folder name

get_sid_path() {
    local dataset=$1
    local sid_type=$2
    local base="${dataset%%_*}"
    
    local sid_folder="$sid_type"
    [[ "$sid_type" == "vqvae" ]] && sid_folder="rvq"
    
    echo "logs/inference/runs/output/${sid_folder}-${base}-128/pickle/merged_predictions_tensor.pt"
}

# ==============================================================================
# Main Configuration Function
# ==============================================================================

# Usage: get_dataset_config <dataset> <sid_type>
# Returns: <total_items>|<embedding_path>|<sid_path>
get_dataset_config() {
    local dataset=$1
    local sid_type=$2
    local base="${dataset%%_*}"
    
    # Validate dataset
    local total_items="${DATASET_ITEM_COUNTS[$base]}"
    [[ -z "$total_items" ]] && return 1
    
    # Get embedding path
    local embedding_path="${DATASET_EMBEDDING_PATHS[$base]}"
    [[ -z "$embedding_path" ]] && return 1
    
    # Get SID path
    local sid_path=$(get_sid_path "$dataset" "$sid_type")
    
    echo "${total_items}|${embedding_path}|${sid_path}"
}

# ==============================================================================
# Export Functions
# ==============================================================================

export -f get_dataset_config
export -f get_sid_path
