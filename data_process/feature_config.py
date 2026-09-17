#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature processing configuration file
Contains all configuration parameters needed for feature post-processing
"""

# Model feature configuration
class ModelFeatureConfig:
    """Model feature configuration class"""

    # Tokens per session: scenario(S) -> intention(I) -> feedback(F)
    action_cnt_in_session = 3

    # Maximum sequence length (40 sessions * 3 tokens)
    max_seq_len = 120

    # Negative sampling configuration
    random_negative_sample_num = 14  # Random negative sampling from full vocabulary
    geographic_negative_sample_num = 50  # Geographic region negative sampling
    negative_sample_num = 64  # random_negative_sample_num + geographic_negative_sample_num

    # Label space of the classification tasks
    travel_mode_class_num = 6  # how
    future_travel_class_num = 49  # when: hour buckets 0..47 plus a ">=2 days" bucket

    # User profile feature configuration
    # Format: "feature_name": [start_index, feature_count]
    u_feature_name_total = {
        "profile_feature_1": [0, 4],
        "profile_feature_2": [4, 2], 
        "profile_feature_3": [6, 9],
        "profile_feature_4": [15, 18],
        "profile_feature_5": [33, 13],
        "profile_feature_6": [46, 1]
    }


# Sequence type mapping dictionary
class SequenceTypeDict:
    """Sequence type dictionary configuration"""

    seq_type_dict = {
        "feedback": 2,
        "intention": 1,
        "scenario": 0,
        "user_profile": 3
    }

    GMODE_LEN = 6


# Create configuration instances
mfc = ModelFeatureConfig()
map_dict = SequenceTypeDict()
