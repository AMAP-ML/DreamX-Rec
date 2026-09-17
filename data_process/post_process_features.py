#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import os
import sys
import csv
import json
import math
import random
import copy
import argparse
from collections import defaultdict

try:
    from .feature_config import mfc, map_dict
except ImportError:
    from feature_config import mfc, map_dict

class FeaturePostProcessor(object):
    def __init__(self, input_file="./output/multi_task_input_seq_all.csv", 
                 output_file="./output/processed_features.csv"):
        self.input_file = input_file
        self.output_file = output_file
        self.status = "train"  # Model training mode
        
    def process_all_data(self):
        """Process all data"""
        print("Starting feature post-processing...")
        print("Input file: {}".format(self.input_file))
        print("Output file: {}".format(self.output_file))
        
        processed_data = []
        
        # Read input data
        with open(self.input_file, 'r') as f:
            # Detect delimiter
            first_line = f.readline()
            if '\t' in first_line:
                delimiter = '\t'
                print("Detected tab delimiter")
            else:
                delimiter = ','
                print("Using comma delimiter")
            
            # Re-read the file
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                try:
                    # Process single sample
                    result = self.process_single_sample(row)
                    if result:
                        processed_data.append(result)
                except Exception as e:
                    print("Error processing sample (user_id={}): {}".format(row.get('user_id', 'unknown'), str(e)))
                    continue
        
        # Save processed results
        self.save_processed_data(processed_data)
        print("Feature post-processing completed! Processed {} samples".format(len(processed_data)))
        
    def process_single_sample(self, in_params):
        """
        :param in_params: Dictionary containing user_id, seq_info, rand_1
        :return: Processed feature dictionary
        """
        # Parse input parameters
        user_id = str(in_params.get('user_id', ''))
        seq_info_json = in_params.get('seq_info', '')
        
        # Debug information
        if not seq_info_json:
            print("Warning: user_id={} has empty seq_info".format(user_id))
            return None
        
        # Parse JSON string to get content
        try:
            in_msg = json.loads(seq_info_json)
            content = in_msg.get('content', {})
        except Exception as e:
            # If parsing fails, try using seq_info_json directly as content
            try:
                content = json.loads(seq_info_json) if isinstance(seq_info_json, str) else {}
            except Exception as e2:
                print("Error: user_id={} JSON parsing failed: {}".format(user_id, str(e2)))
                return None
        
        # Check required fields
        seq_info = content.get("seq_info")
        if not seq_info:
            print("Warning: user_id={} has no seq_info field in content".format(user_id))
            return None
        
        # Add user ID to content
        content['user_id'] = user_id
        
        # Generate features
        try:
            model_feature = self.generate_model_feature(content)
            model_feature['user_id'] = user_id
            return model_feature
        except Exception as e:
            print("Error: user_id={} feature generation failed: {}".format(user_id, str(e)))
            return None
    
    def generate_model_feature(self, in_msg):
        seq_info = in_msg.get("seq_info")
        
        # ******************** step1: Process sequence features ********************
        
        # Apply to all tokens in sequence:
        type_index = []
        type_index_detail = []
        action_list_time_feature = []
        
        # Apply only to S token:
        action_list_exp_geographic = []
        action_list_exp_user_loc_administrative_region_index = []
        action_list_exp_source_condition_index = []
        
        # Apply only to I token (poi static info):
        action_list_poi_id = []
        action_list_poi_geographic = []
        action_list_poi_base_score = []
        action_list_poi_category = []
        action_list_poi_administrative_region = []
        
        # Apply only to F token:
        action_list_travel_mode = []
        
        # poi labels: "where" on S, "via" on I, together with their negatives
        label_poi_id = []
        label_poi_geographic = []
        label_poi_score = []
        label_poi_category = []
        label_poi_administrative_region = []
        label_neg_poi_id = []
        label_neg_poi_geographic = []
        label_neg_poi_score = []
        label_neg_poi_category = []
        label_neg_poi_administrative_region = []
        # "how" and "when" labels, both on S
        label_travel_mode = []
        label_future_travel = []
        
        seq_info_list = seq_info.split("&")
        # Filter out empty actions
        seq_info_list = [x for x in seq_info_list if x.strip()]
        seq_info_list_sorted = sorted(seq_info_list, key=lambda x: int(x.split("|")[0]) if x.split("|")[0] else 0, reverse=True)
        
        for i in range(len(seq_info_list_sorted)):
            # Parse features in input sequence:
            # 0:timestamp, 1:action_type, 2:geographic_id, 
            # 3:poi_id, 4:target_poi_geographic_id, 5:target_poi_normalized_score, 6:target_poi_category_id, 7:target_poi_administrative_region_id,
            # 8:administrative_region_id, 9:weather, 10:travel_mode, 11:via_info,
            # 12:negative_samples, 13:geographic_negative_samples
            
            cur_index = i
            cur_action_list = seq_info_list_sorted[cur_index].split("|")
            
            # Fill missing fields
            while len(cur_action_list) < 15:
                cur_action_list.append("")
                
            cur_timestamp = cur_action_list[0]
            cur_time_feature = int(round(int(cur_timestamp)/1000, 0)) if cur_timestamp else 0
            cur_action_type_index = self.str_to_int_index(cur_action_list[1])
            cur_geographic_id = int(cur_action_list[2]) if cur_action_list[2] else -1
            cur_target_poi_index = int(cur_action_list[3]) if cur_action_list[3] else -1
            cur_target_poi_geographic_id_index = int(cur_action_list[4]) if cur_action_list[4] else -1
            cur_target_poi_normalized_score = float(cur_action_list[5]) if (cur_action_list[5] and float(cur_action_list[5]) >= 0 and float(cur_action_list[5]) <= 1) else 0
            cur_poi_category_id_path_index = self.str_to_int_index(cur_action_list[6])
            target_poi_administrative_region_id_index = self.str_to_int_index(cur_action_list[7])
            administrative_region_id_index = self.str_to_int_index(cur_action_list[8])
            weather_index = self.str_to_int_index(cur_action_list[9])
            label_travel_mode_index = int(cur_action_list[10]) if (cur_action_list[10] and cur_action_list[10] != "") else -1
            via_info = cur_action_list[11]
            
            label_future_travel_gap = cur_action_list[14] if cur_action_list[14] else "0"
            label_future_travel_gap_index = self.get_label_future_travel_gap(label_future_travel_gap)
            
            # Ground truth of the next session, consumed by the "via" task on I:
            next_poi_index, next_poi_geographic_index, next_poi_base_score, \
                next_poi_category, next_poi_administrative_region, \
                    next_neg_sample_poi, next_neg_sample_geographic, next_neg_sample_score, \
                        next_neg_sample_category, next_neg_sample_administrative_region = \
                            self.get_label_poi_info_list(seq_info_list_sorted, i, self.status)
            
            # If current action has via point, put via point in I position and label of S; otherwise put destination
            if via_info != "":
                via_info_list = via_info.split(",")
                if len(via_info_list) >= 5:
                    cur_target_poi_index = int(via_info_list[0]) if via_info_list[0] else -1
                    cur_target_poi_geographic_id_index = int(via_info_list[1]) if via_info_list[1] else -1
                    cur_target_poi_normalized_score = float(via_info_list[2]) if via_info_list[2] else 0
                    cur_poi_category_id_path = self.str_to_int_index(via_info_list[3])
                    cur_poi_category_id_path_index = cur_poi_category_id_path
                    target_poi_administrative_region_id_index = self.str_to_int_index(via_info_list[4])
            
            # Negatives of the current session, consumed by the "where" task on S:
            if self.status == "train":
                cur_random_negative_sample = cur_action_list[12]
                cur_geographic_negative_sample = cur_action_list[13]
                cur_random_neg_sample_poi, cur_random_neg_sample_geographic, cur_random_neg_sample_score, \
                        cur_random_neg_sample_category, cur_random_neg_sample_administrative_region = \
                    self.get_negative_sample_feature(
                        cur_random_negative_sample, 
                        mfc.random_negative_sample_num
                    ) 
                cur_geographic_neg_sample_poi, cur_geographic_neg_sample_geographic, cur_geographic_neg_sample_score, \
                        cur_geographic_neg_sample_category, cur_geographic_neg_sample_administrative_region = \
                    self.get_negative_sample_feature(
                        cur_geographic_negative_sample, 
                        mfc.geographic_negative_sample_num
                    )
                cur_neg_sample_poi = cur_random_neg_sample_poi + cur_geographic_neg_sample_poi
                cur_neg_sample_geographic = cur_random_neg_sample_geographic + cur_geographic_neg_sample_geographic
                cur_neg_sample_score = cur_random_neg_sample_score + cur_geographic_neg_sample_score
                cur_neg_sample_category = cur_random_neg_sample_category + cur_geographic_neg_sample_category
                cur_neg_sample_administrative_region = cur_random_neg_sample_administrative_region + cur_geographic_neg_sample_administrative_region
        
            # Record (chronological order) S -> I -> F, stored in reverse order F, I, S
            # Apply to all tokens in sequence, 3 tokens:
            type_index += [map_dict.seq_type_dict["feedback"], map_dict.seq_type_dict["intention"], map_dict.seq_type_dict["scenario"]]
            type_index_detail += [cur_action_type_index, -1, -1]
            action_list_time_feature += [cur_time_feature] * 3
            
            # Apply only to S token:
            action_list_exp_geographic += [-1, -1, cur_geographic_id]
            action_list_exp_user_loc_administrative_region_index += [-1, -1, administrative_region_id_index]
            action_list_exp_source_condition_index += [-1, -1, weather_index]
            
            # Apply only to I token:
            action_list_poi_id += [-1, cur_target_poi_index, -1]
            action_list_poi_geographic += [-1, cur_target_poi_geographic_id_index, -1]
            action_list_poi_base_score += [0, cur_target_poi_normalized_score, 0]
            action_list_poi_category += [-1, cur_poi_category_id_path_index, -1]
            action_list_poi_administrative_region += [-1, target_poi_administrative_region_id_index, -1]
            
            # Apply only to F token:
            action_list_travel_mode += [label_travel_mode_index, -1, -1]
            
            # Multi-task labels and negative samples:
            if self.status == "train":
                # **************** One label array carries "via" on I and "where" on S ****************
                label_poi_id += [-1, next_poi_index, cur_target_poi_index]
                label_poi_geographic += [-1, next_poi_geographic_index, cur_target_poi_geographic_id_index]
                label_poi_score += [0.0, next_poi_base_score, cur_target_poi_normalized_score]
                label_poi_category += [-1, next_poi_category, cur_poi_category_id_path_index]
                label_poi_administrative_region += [-1, next_poi_administrative_region, target_poi_administrative_region_id_index]
                
                label_neg_poi_id += [-1] * mfc.negative_sample_num + next_neg_sample_poi + cur_neg_sample_poi
                label_neg_poi_geographic += [-1] * mfc.negative_sample_num + next_neg_sample_geographic + cur_neg_sample_geographic
                label_neg_poi_score += [0.0] * mfc.negative_sample_num + next_neg_sample_score + cur_neg_sample_score
                label_neg_poi_category += [-1] * mfc.negative_sample_num + next_neg_sample_category + cur_neg_sample_category
                label_neg_poi_administrative_region += [-1] * mfc.negative_sample_num + next_neg_sample_administrative_region + cur_neg_sample_administrative_region
                
                # **************** "how" and "when" are query-style tasks on S; their observed values sit on F, which S cannot attend to ****************
                label_travel_mode += [-1, -1, label_travel_mode_index]
                label_future_travel += [-1, -1, label_future_travel_gap_index]
        
        # ******************** step2: Process user profile ******************** 
        u_feature_id = self.get_user_profile(in_msg)
        
        # ******************** step3: Concatenate history sequence and user profile ******************** 
        len_orig_seq = min(mfc.max_seq_len, len(type_index))
        len_profile = len(mfc.u_feature_name_total)
        
        # Apply to all tokens in sequence:
        type_index = type_index[:mfc.max_seq_len] + [map_dict.seq_type_dict["user_profile"]] * len_profile
        type_index_detail = type_index_detail[:mfc.max_seq_len] + [-1] * len_profile
        action_list_time_feature = action_list_time_feature[:mfc.max_seq_len] + [-1] * len_profile
        
        # Apply only to S token:
        action_list_exp_geographic = action_list_exp_geographic[:mfc.max_seq_len] + [-1] * len_profile
        action_list_exp_user_loc_administrative_region_index = action_list_exp_user_loc_administrative_region_index[:mfc.max_seq_len] + [-1] * len_profile
        action_list_exp_source_condition_index = action_list_exp_source_condition_index[:mfc.max_seq_len] + [-1] * len_profile
        
        # Apply only to I token (poi static info):
        action_list_poi_id = action_list_poi_id[:mfc.max_seq_len] + [-1] * len_profile
        action_list_poi_geographic = action_list_poi_geographic[:mfc.max_seq_len] + [-1] * len_profile
        action_list_poi_base_score = action_list_poi_base_score[:mfc.max_seq_len] + [0.0] * len_profile
        action_list_poi_category = action_list_poi_category[:mfc.max_seq_len] + [-1] * len_profile
        action_list_poi_administrative_region = action_list_poi_administrative_region[:mfc.max_seq_len] + [-1] * len_profile
        
        # Apply only to F token:
        action_list_travel_mode = action_list_travel_mode[:mfc.max_seq_len] + [-1] * len_profile
        
        # Apply only to U token: 
        u_feature_id = [-1] * len_orig_seq + u_feature_id
        
        if self.status == "train":
            # poi labels and their negatives:
            label_poi_id = label_poi_id[:mfc.max_seq_len] + [-1] * len_profile
            label_poi_geographic = label_poi_geographic[:mfc.max_seq_len] + [-1] * len_profile
            label_poi_score = label_poi_score[:mfc.max_seq_len] + [0.0] * len_profile
            label_poi_category = label_poi_category[:mfc.max_seq_len] + [-1] * len_profile
            label_poi_administrative_region = label_poi_administrative_region[:mfc.max_seq_len] + [-1] * len_profile
        
            label_neg_poi_id = label_neg_poi_id[:mfc.max_seq_len*mfc.negative_sample_num] + [-1] * mfc.negative_sample_num * len_profile
            label_neg_poi_geographic = label_neg_poi_geographic[:mfc.max_seq_len*mfc.negative_sample_num] + [-1] * mfc.negative_sample_num * len_profile
            label_neg_poi_score = label_neg_poi_score[:mfc.max_seq_len*mfc.negative_sample_num] + [0.0] * mfc.negative_sample_num * len_profile
            label_neg_poi_category = label_neg_poi_category[:mfc.max_seq_len*mfc.negative_sample_num] + [-1] * mfc.negative_sample_num * len_profile
            label_neg_poi_administrative_region = label_neg_poi_administrative_region[:mfc.max_seq_len*mfc.negative_sample_num] + [-1] * mfc.negative_sample_num * len_profile
            
            # "how" label:
            label_travel_mode = label_travel_mode[:mfc.max_seq_len] + [-1] * len_profile
            # "when" label:
            label_future_travel = label_future_travel[:mfc.max_seq_len] + [-1] * len_profile
        
        # ******************** step5: Organize final return result ******************** 
        res = {
            # Apply to all tokens in sequence:
            "type_index": type_index,
            "type_index_detail": type_index_detail,
            "action_list_time_feature": action_list_time_feature,
            
            # Apply only to S token:
            "action_list_exp_geographic": action_list_exp_geographic,
            "action_list_exp_user_loc_administrative_region_index": action_list_exp_user_loc_administrative_region_index,
            "action_list_exp_source_condition_index": action_list_exp_source_condition_index,
            
            # Apply only to I token (poi static info):
            "action_list_poi_id": action_list_poi_id,
            "action_list_poi_geographic": action_list_poi_geographic,
            "action_list_poi_base_score": action_list_poi_base_score,
            "action_list_poi_category": action_list_poi_category,
            "action_list_poi_administrative_region": action_list_poi_administrative_region,
            
            # Apply only to F token:
            "action_list_travel_mode": action_list_travel_mode,
            
            # Apply only to U token:
            "u_feature_id": u_feature_id,
        }
        
        if self.status == "train":
            res_label_neg = {
                # "where" label on S, "via" label on I, plus shared negatives:
                "label_poi_id": label_poi_id,
                "label_poi_geographic": label_poi_geographic,
                "label_poi_score": label_poi_score,
                "label_poi_category": label_poi_category,
                "label_poi_administrative_region": label_poi_administrative_region,
                "label_neg_poi_id": label_neg_poi_id,
                "label_neg_poi_geographic": label_neg_poi_geographic,
                "label_neg_poi_score": label_neg_poi_score,
                "label_neg_poi_category": label_neg_poi_category,
                "label_neg_poi_administrative_region": label_neg_poi_administrative_region,
                        
                # "how" label on S:
                "label_travel_mode": label_travel_mode,
                # "when" label on S:
                "label_future_travel": label_future_travel
            }
            res.update(res_label_neg)
        
        return res
    
    def str_to_int_index(self, value_str):
        """Convert string to integer index"""
        if value_str == "" or value_str == "-1":
            return -1
        try:
            return int(value_str)
        except:
            return -1
    
    def get_label_poi_info_list(self, seq_info_list_sorted, cur_index, status):
        """Find label for I position in SQIF session extracted from current action"""
        if cur_index == 0: # Last planning, no ground truth
            next_poi_index = -1
            next_poi_geographic_index = -1
            next_poi_base_score = 0.0
            next_poi_category = -1
            next_poi_administrative_region = -1
            next_random_negative_sample = ""
            next_geographic_negative_sample = ""
        else:
            next_index = cur_index - 1
            next_action_list = seq_info_list_sorted[next_index].split('|')
            
            # Fill missing fields
            while len(next_action_list) < 14:
                next_action_list.append("")
                
            next_via_info = next_action_list[11]
            if next_via_info != "":
                next_via_info_list = next_via_info.split(",")
                if len(next_via_info_list) >= 5:
                    next_poi_index = int(next_via_info_list[0]) if next_via_info_list[0] else -1
                    next_poi_geographic_index = int(next_via_info_list[1]) if next_via_info_list[1] else -1
                    next_poi_base_score = float(next_via_info_list[2]) if next_via_info_list[2] else 0.0
                    next_poi_category = self.str_to_int_index(next_via_info_list[3])
                    next_poi_administrative_region = self.str_to_int_index(next_via_info_list[4])
            else:
                next_poi_index = int(next_action_list[3]) if next_action_list[3] else -1
                next_poi_geographic_index = int(next_action_list[4]) if next_action_list[4] else -1
                next_poi_base_score = float(next_action_list[5]) if next_action_list[5] else 0.0
                next_poi_category = self.str_to_int_index(next_action_list[6])
                next_poi_administrative_region = self.str_to_int_index(next_action_list[7])
            
            if status == "train":
                next_random_negative_sample = next_action_list[12]
                next_geographic_negative_sample = next_action_list[13]
            else:
                next_random_negative_sample = ""
                next_geographic_negative_sample = ""
        
        # Process negative samples:
        next_random_neg_sample_poi, next_random_neg_sample_geographic, next_random_neg_sample_score, \
                next_random_neg_sample_category, next_random_neg_sample_administrative_region = \
            self.get_negative_sample_feature(
                next_random_negative_sample, 
                mfc.random_negative_sample_num
            ) 
        next_geographic_neg_sample_poi, next_geographic_neg_sample_geographic, next_geographic_neg_sample_score, \
                next_geographic_neg_sample_category, next_geographic_neg_sample_administrative_region = \
            self.get_negative_sample_feature(
                next_geographic_negative_sample, 
                mfc.geographic_negative_sample_num
            )
        next_neg_sample_poi = next_random_neg_sample_poi + next_geographic_neg_sample_poi
        next_neg_sample_geographic = next_random_neg_sample_geographic + next_geographic_neg_sample_geographic
        next_neg_sample_score = next_random_neg_sample_score + next_geographic_neg_sample_score
        next_neg_sample_category = next_random_neg_sample_category + next_geographic_neg_sample_category
        next_neg_sample_administrative_region = next_random_neg_sample_administrative_region + next_geographic_neg_sample_administrative_region
        
        return next_poi_index, next_poi_geographic_index, next_poi_base_score, \
            next_poi_category, next_poi_administrative_region, \
            next_neg_sample_poi, next_neg_sample_geographic, next_neg_sample_score, \
            next_neg_sample_category, next_neg_sample_administrative_region
    
    def get_label_future_travel_gap(self, label_future_travel_gap):
        """Process departure time gap label"""
        int_time_gap = int(label_future_travel_gap) / 1000  # Convert ms to seconds
        hour_gap = math.floor(int_time_gap / 3600)
        if 0 <= hour_gap and hour_gap <= 47:
            return hour_gap
        else:  # Unify 2+ days as 2 days, or negative (-1 = no future travel)
            return 48
    
    def get_negative_sample_feature(self, negative_sample, sample_num):
        """Process negative sampling features"""
        neg_sample_poi = [-1] * sample_num
        neg_sample_geographic = [-1] * sample_num
        neg_sample_score = [0] * sample_num
        neg_sample_category = [-1] * sample_num
        neg_sample_administrative_region = [-1] * sample_num
        
        if negative_sample != "": 
            negative_sample_list = negative_sample.split(";")
            for i in range(min(len(negative_sample_list), sample_num)):
                cur_sample_info_list = negative_sample_list[i].split(",")
                if len(cur_sample_info_list) >= 5:
                    neg_sample_poi[i] = int(cur_sample_info_list[0]) if cur_sample_info_list[0] else -1
                    neg_sample_geographic[i] = int(cur_sample_info_list[1]) if cur_sample_info_list[1] else -1
                    neg_sample_score[i] = float(cur_sample_info_list[2]) if cur_sample_info_list[2] else 0.0
                    neg_sample_category[i] = self.str_to_int_index(cur_sample_info_list[3])
                    neg_sample_administrative_region[i] = self.str_to_int_index(cur_sample_info_list[4])
        return neg_sample_poi, neg_sample_geographic, neg_sample_score, neg_sample_category, neg_sample_administrative_region
    
    def get_user_profile(self, in_msg):
        """Process user profile features"""
        u_feature_id = []
        for profile_feature_name in mfc.u_feature_name_total:
            feature_value = in_msg.get(profile_feature_name, "")
            if feature_value == "" or feature_value == "-1" or feature_value == -1:
                feature_value_index = -1
            else:
                feature_value_index = int(feature_value)
                accu_feature_cnt = mfc.u_feature_name_total[profile_feature_name][0]
                feature_value_index += accu_feature_cnt
            u_feature_id.append(feature_value_index)
        return u_feature_id
    
    def save_processed_data(self, processed_data):
        """Save processed data as standard CSV format"""
        if not processed_data:
            print("No data to save")
            return
            
        # Create output directory
        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Collect all possible feature column names
        if processed_data:
            all_columns = ['user_id']  # user_id as first column
            # Add all feature column names
            sample_features = processed_data[0]
            feature_columns = [key for key in sample_features.keys() if key != 'user_id']
            all_columns.extend(sorted(feature_columns))  # Sort to maintain consistency
        else:
            all_columns = ['user_id']
        
        # Save as standard CSV format
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(all_columns)
            
            # Write data rows
            for sample in processed_data:
                row = []
                for col in all_columns:
                    if col == 'user_id':
                        # Write user_id field directly
                        row.append(sample.get('user_id', ''))
                    else:
                        # Other fields are lists, convert to pure array format string
                        value = sample.get(col, [])
                        if isinstance(value, list):
                            # Convert to unquoted array format: [-1,-1,8,...]
                            array_str = '[' + ','.join(map(str, value)) + ']'
                            row.append(array_str)
                        else:
                            # If not a list, convert to single-element array
                            array_str = '[' + str(value) + ']'
                            row.append(array_str)
                writer.writerow(row)
        
        print("Processed results saved to: {}".format(self.output_file))
        
        # Display statistics
        print("\nProcessing statistics:")
        print("- Total samples: {}".format(len(processed_data)))
        print("- Feature columns: {}".format(len(all_columns)))
        if processed_data:
            sample = processed_data[0]
            print("- Feature dimension examples:")
            for key, value in sample.items():
                if key != 'user_id' and isinstance(value, list):
                    print("  {}: {}".format(key, len(value)))
        
        # Verify output file format
        print("\nVerifying output file format:")
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                print("- Header line: {}".format(lines[0].strip()))
            if len(lines) > 1:
                data_fields = lines[1].strip().split(',')
                print("- First data field count: {}".format(len(data_fields)))
                print("- First data sample: {}".format(lines[1].strip()[:100] + "..." if len(lines[1]) > 100 else lines[1].strip()))

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Feature post-processing program')
    parser.add_argument('--input-file', '-i', default='./output/multi_task_input_seq_all.csv',
                       help='Input file path (default: ./output/multi_task_input_seq_all.csv)')
    parser.add_argument('--output-file', '-o', default='./output/processed_features.csv',
                       help='Output file path (default: ./output/processed_features.csv)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print("Error: Input file {} does not exist".format(args.input_file))
        sys.exit(1)
    
    print("=" * 50)
    print("Feature Post-processing Program")
    print("=" * 50)
    print("Input file: {}".format(args.input_file))
    print("Output file: {}".format(args.output_file))
    print("=" * 50)
    
    # Create processor and execute
    processor = FeaturePostProcessor(args.input_file, args.output_file)
    processor.process_all_data()

if __name__ == "__main__":
    main()