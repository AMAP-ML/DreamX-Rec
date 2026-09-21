"""Processed IntTravel feature adapter for the recommendation example.

The IntTravel data processor emits canonical newest-first ``[F, I, S]``
sessions. The public label attached to S is returned as session-level target
supervision rather than being treated as an input feature.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch.utils.data import Dataset

from .config import (
    FEEDBACK_TYPE,
    INTENTION_TYPE,
    PROFILE_TYPE,
    SCENARIO_TYPE,
    RecommendationConfig,
)


MODEL_FEATURE_NAMES = (
    "type_index",
    "type_index_detail",
    "action_list_exp_geographic",
    "action_list_exp_user_loc_administrative_region_index",
    "action_list_exp_source_condition_index",
    "action_list_poi_id",
    "action_list_poi_geographic",
    "action_list_poi_base_score",
    "action_list_poi_category",
    "action_list_poi_administrative_region",
    "action_list_travel_mode",
    "u_feature_id",
    "label_poi_id",
    "label_poi_geographic",
    "label_poi_score",
    "label_poi_category",
    "label_poi_administrative_region",
    "label_neg_poi_id",
    "label_neg_poi_geographic",
    "label_neg_poi_score",
    "label_neg_poi_category",
    "label_neg_poi_administrative_region",
)


def load_public_features(processed_file: str | Path) -> list[dict]:
    """Load the feature CSV emitted by ``FeaturePostProcessor``."""

    path = Path(processed_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"processed recommendation data not found: {path}. "
            "Run data_processor.py and post_process_features.py first."
        )

    features = []
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing = set(MODEL_FEATURE_NAMES) - fieldnames
        if missing:
            raise ValueError(
                f"processed recommendation data is missing columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                features.append(
                    {name: json.loads(row[name]) for name in MODEL_FEATURE_NAMES}
                )
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid feature array in {path} at row {row_number}"
                ) from error
    return features


def _negative_rows(values: list, token_count: int, negatives_per_token: int) -> list[list]:
    return [
        values[index * negatives_per_token : (index + 1) * negatives_per_token]
        for index in range(token_count)
    ]


def _empty_tokens(config: RecommendationConfig) -> Dict[str, list]:
    length = config.max_sequence_length
    return {
        "type_id": [-1] * length,
        "detail_id": [-1] * length,
        "scenario_geographic_id": [-1] * length,
        "scenario_region_id": [-1] * length,
        "weather_id": [-1] * length,
        "poi_id": [-1] * length,
        "poi_geographic_id": [-1] * length,
        "poi_score": [0.0] * length,
        "poi_category_id": [-1] * length,
        "poi_region_id": [-1] * length,
        "travel_mode_id": [-1] * length,
        "profile_id": [-1] * length,
        "valid": [False] * length,
    }


def adapt_public_feature(
    feature: Mapping[str, list],
    config: RecommendationConfig | None = None,
) -> Dict[str, object]:
    cfg = config or RecommendationConfig()
    token_count = len(feature["type_index"])
    action_token_count = token_count - cfg.num_profile_tokens
    if action_token_count % cfg.tokens_per_session:
        raise ValueError("public action sequence is not divisible into [F, I, S] sessions")
    session_count = action_token_count // cfg.tokens_per_session
    if session_count > cfg.max_sessions:
        raise ValueError("public sequence contains more sessions than the model limit")

    tokens = _empty_tokens(cfg)
    target_positions = [-1] * cfg.max_sessions
    target_valid = [False] * cfg.max_sessions

    # Public fields use descriptive names and are mapped to the model's
    # contiguous [F, I, S] token layout.
    for session in range(session_count):
        public_base = session * cfg.tokens_per_session
        public_f, public_i, public_s = public_base, public_base + 1, public_base + 2
        model_base = session * cfg.tokens_per_session
        model_f, model_i, model_s = (
            model_base,
            model_base + 1,
            model_base + 2,
        )

        tokens["type_id"][model_f] = FEEDBACK_TYPE
        tokens["type_id"][model_i] = INTENTION_TYPE
        tokens["type_id"][model_s] = SCENARIO_TYPE
        tokens["detail_id"][model_f] = feature["type_index_detail"][public_f]
        tokens["detail_id"][model_i] = feature["type_index_detail"][public_i]
        tokens["detail_id"][model_s] = feature["type_index_detail"][public_s]

        tokens["scenario_geographic_id"][model_s] = feature["action_list_exp_geographic"][public_s]
        tokens["scenario_region_id"][model_s] = feature[
            "action_list_exp_user_loc_administrative_region_index"
        ][public_s]
        tokens["weather_id"][model_s] = feature["action_list_exp_source_condition_index"][public_s]

        tokens["poi_id"][model_i] = feature["action_list_poi_id"][public_i]
        tokens["poi_geographic_id"][model_i] = feature["action_list_poi_geographic"][public_i]
        tokens["poi_score"][model_i] = feature["action_list_poi_base_score"][public_i]
        tokens["poi_category_id"][model_i] = feature["action_list_poi_category"][public_i]
        tokens["poi_region_id"][model_i] = feature["action_list_poi_administrative_region"][public_i]
        tokens["travel_mode_id"][model_f] = feature["action_list_travel_mode"][public_f]

        for position in (model_f, model_i, model_s):
            tokens["valid"][position] = True
        target_positions[session] = model_s
        target_valid[session] = True

    # Profiles remain after the valid action sequence, matching the processed
    # feature layout. They are visible as the oldest keys under the reverse-time
    # causal mask.
    public_profile_start = action_token_count
    model_profile_start = session_count * cfg.tokens_per_session
    for offset in range(cfg.num_profile_tokens):
        position = model_profile_start + offset
        tokens["type_id"][position] = PROFILE_TYPE
        tokens["detail_id"][position] = feature["type_index_detail"][public_profile_start + offset]
        tokens["profile_id"][position] = feature["u_feature_id"][public_profile_start + offset]
        tokens["valid"][position] = True

    negative_id = _negative_rows(
        feature["label_neg_poi_id"], token_count, cfg.num_negative_samples
    )
    negative_geographic = _negative_rows(
        feature["label_neg_poi_geographic"], token_count, cfg.num_negative_samples
    )
    negative_score = _negative_rows(
        feature["label_neg_poi_score"], token_count, cfg.num_negative_samples
    )
    negative_category = _negative_rows(
        feature["label_neg_poi_category"], token_count, cfg.num_negative_samples
    )
    negative_region = _negative_rows(
        feature["label_neg_poi_administrative_region"], token_count, cfg.num_negative_samples
    )

    labels = {
        "positive_poi_id": [-1] * cfg.max_sessions,
        "positive_geographic_id": [-1] * cfg.max_sessions,
        "positive_score": [0.0] * cfg.max_sessions,
        "positive_category_id": [-1] * cfg.max_sessions,
        "positive_region_id": [-1] * cfg.max_sessions,
        "negative_poi_id": [[-1] * cfg.num_negative_samples for _ in range(cfg.max_sessions)],
        "negative_geographic_id": [[-1] * cfg.num_negative_samples for _ in range(cfg.max_sessions)],
        "negative_score": [[0.0] * cfg.num_negative_samples for _ in range(cfg.max_sessions)],
        "negative_category_id": [[-1] * cfg.num_negative_samples for _ in range(cfg.max_sessions)],
        "negative_region_id": [[-1] * cfg.num_negative_samples for _ in range(cfg.max_sessions)],
    }
    for session in range(session_count):
        public_s = session * cfg.tokens_per_session + 2
        labels["positive_poi_id"][session] = feature["label_poi_id"][public_s]
        labels["positive_geographic_id"][session] = feature["label_poi_geographic"][public_s]
        labels["positive_score"][session] = feature["label_poi_score"][public_s]
        labels["positive_category_id"][session] = feature["label_poi_category"][public_s]
        labels["positive_region_id"][session] = feature["label_poi_administrative_region"][public_s]
        labels["negative_poi_id"][session] = negative_id[public_s]
        labels["negative_geographic_id"][session] = negative_geographic[public_s]
        labels["negative_score"][session] = negative_score[public_s]
        labels["negative_category_id"][session] = negative_category[public_s]
        labels["negative_region_id"][session] = negative_region[public_s]

    float_token_keys = {"poi_score"}
    bool_token_keys = {"valid"}
    tensor_tokens = {
        key: torch.tensor(
            value,
            dtype=torch.float32 if key in float_token_keys else torch.bool if key in bool_token_keys else torch.long,
        )
        for key, value in tokens.items()
    }
    float_label_keys = {"positive_score", "negative_score"}
    tensor_labels = {
        key: torch.tensor(value, dtype=torch.float32 if key in float_label_keys else torch.long)
        for key, value in labels.items()
    }
    return {
        "tokens": tensor_tokens,
        "labels": tensor_labels,
        "target_positions": torch.tensor(target_positions, dtype=torch.long),
        "target_valid": torch.tensor(target_valid, dtype=torch.bool),
    }


class IntTravelRecommendationDataset(Dataset):
    def __init__(
        self,
        processed_file: str | Path,
        config: RecommendationConfig | None = None,
    ):
        self.config = config or RecommendationConfig()
        features = load_public_features(processed_file)
        self.samples = [adapt_public_feature(feature, self.config) for feature in features]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.samples[index]

    def embedding_vocabulary(self) -> tuple[set[int], set[int]]:
        poi_ids: set[int] = set()
        geographic_ids: set[int] = set()
        for sample in self.samples:
            tokens = sample["tokens"]
            labels = sample["labels"]
            for tensor in (
                tokens["poi_id"],
                labels["positive_poi_id"],
                labels["negative_poi_id"],
            ):
                poi_ids.update(int(value) for value in tensor.reshape(-1).tolist() if value >= 0)
            for tensor in (
                tokens["scenario_geographic_id"],
                tokens["poi_geographic_id"],
                labels["positive_geographic_id"],
                labels["negative_geographic_id"],
            ):
                geographic_ids.update(int(value) for value in tensor.reshape(-1).tolist() if value >= 0)
        return poi_ids, geographic_ids


def move_to(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to(item, device) for key, item in value.items()}
    return value
