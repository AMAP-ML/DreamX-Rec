#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing main program - Python2.7 compatible version
Processing training data generation pipeline for multi-task learning tasks
"""

import os
import sys
import csv
import json
import random
import time
import copy
import argparse
from collections import defaultdict

# Compatible with Python2.7 dictionary ordering
try:
    from collections import OrderedDict
except ImportError:
    # Versions before Python 2.7 don't have OrderedDict
    OrderedDict = dict

class DataProcessor(object):
    def __init__(self, data_dir='./raw_data', output_dir='./output', 
                 input_files=None, required_columns=None, max_actions=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_actions = max_actions
        
        # Default input file configuration
        self.default_files = {
            'user_action': 'user_action.csv',
            'user_profile': 'user_profile.csv', 
            'poi_info': 'poi_info.csv',
            'poi_info_groupby_geographic': 'poi_info_groupby_geographic.csv'
        }
        
        # Default required column configuration (based on actual table structure)
        self.default_columns = {
            'user_action': ['user_id', 'timestamp', 'action_type', 'poi_id', 
                           'geographic_id', 'administrative_region_id', 'weather', 'travel_mode', 'via_poi_id'],
            'user_profile': ['user_id', 'profile_feature_1', 'profile_feature_2', 'profile_feature_3',
                            'profile_feature_4', 'profile_feature_5', 'profile_feature_6'],
            'poi_info': ['poi_id', 'normalized_score', 'geographic_id', 'category_id', 'administrative_region_id'],
            'poi_info_groupby_geographic': ['geographic_id', 'poi_info']
        }
        
        # Use custom configuration or default configuration
        self.input_files = input_files if input_files else self.default_files
        self.required_columns = required_columns if required_columns else self.default_columns
        
        # Data storage
        self.user_actions = {}  # user_id -> actions list
        self.user_profiles = {}  # user_id -> profile info
        self.poi_info = {}  # poi_id -> poi info
        self.geographic_poi_info = {}  # geographic_id -> poi_info string
        
        # Configuration parameters
        self.max_poi_index = 7291872  # Total POI count
        self.neg_sample_num = 14  # Random negative sampling count
        self.tile_sample_num = 50  # Hard negative sampling count
        self.max_seq_length = 40  # Maximum sequence length
    
    def check_columns(self, filepath, table_name):
        """Check if file column names meet requirements"""
        try:
            with open(filepath, 'r') as f:
                # Read first line to get column names
                first_line = f.readline().strip()
                if '\t' in first_line:
                    actual_columns = first_line.split('\t')
                else:
                    actual_columns = first_line.split(',')
                
                # Get expected column names
                expected_columns = self.required_columns.get(table_name, [])
                
                print("Checking {} file columns:".format(table_name))
                print("  Expected columns: {}".format(expected_columns))
                print("  Actual columns: {}".format(actual_columns))
                
                # Check for missing columns
                missing_columns = [col for col in expected_columns if col not in actual_columns]
                extra_columns = [col for col in actual_columns if col not in expected_columns]
                
                if missing_columns:
                    print("  ❌ Missing columns: {}".format(missing_columns))
                    return False
                
                if extra_columns:
                    print("  ⚠️  Extra columns: {}".format(extra_columns))
                
                print("  ✅ Column check passed")
                return True
                
        except Exception as e:
            print("  ❌ Error checking file {}: {}".format(filepath, str(e)))
            return False
        
    def load_data(self, check_columns=True):
        """Load all raw data"""
        print("Starting to load data...")
        
        # Column name check
        if check_columns:
            print("\n=== Column Check ===")
            all_passed = True
            for table_name, filename in self.input_files.items():
                filepath = os.path.join(self.data_dir, filename)
                if not os.path.exists(filepath):
                    print("⚠️  File does not exist, skipping check: {}".format(filepath))
                    continue
                
                if not self.check_columns(filepath, table_name):
                    all_passed = False
            
            if not all_passed:
                print("\n❌ Column check failed, please fix and run again")
                return False
            
            print("✅ All column checks passed\n")
        
        # Load user action data
        self._load_user_actions()
        
        # Load user profile data
        self._load_user_profiles()
        
        # Load POI info data
        self._load_poi_info()
        
        # Load POI info aggregated by geographic ID
        self._load_geographic_poi_info()
        
        print("Data loading completed!")
        print("User action data count:", sum(len(actions) for actions in self.user_actions.values()))
        print("User profile data count:", len(self.user_profiles))
        print("POI info data count:", len(self.poi_info))
        print("Geographic info data count:", len(self.geographic_poi_info))
        
        return True
        
    def _load_user_actions(self):
        """Load user action data"""
        filename = self.input_files['user_action']
        filepath = os.path.join(self.data_dir, filename)
        loaded = 0
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                user_id = row['user_id']
                if user_id not in self.user_actions:
                    self.user_actions[user_id] = []
                
                # Convert data types (based on actual table structure)
                action_data = {
                    'user_id': user_id,
                    'timestamp': int(row['timestamp']),
                    'action_type': row['action_type'],
                    'poi_id': row['poi_id'],
                    'geographic_id': row.get('geographic_id', ''),
                    'administrative_region_id': row.get('administrative_region_id', ''),
                    'weather': row.get('weather', ''),
                    'travel_mode': row.get('travel_mode', ''),
                    'via_poi_id': row.get('via_poi_id', '') if row.get('via_poi_id') else ""
                }
                self.user_actions[user_id].append(action_data)
                loaded += 1
                if self.max_actions is not None and loaded >= self.max_actions:
                    break
            
            # Sort by timestamp
            for user_id in self.user_actions:
                self.user_actions[user_id].sort(key=lambda x: x['timestamp'])
    
    def _load_user_profiles(self):
        """Load user profile data"""
        filename = self.input_files['user_profile']
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                user_id = row['user_id']
                self.user_profiles[user_id] = {
                    'profile_feature_1': row['profile_feature_1'] if row['profile_feature_1'] else "",
                    'profile_feature_2': row['profile_feature_2'] if row['profile_feature_2'] else "",
                    'profile_feature_3': row['profile_feature_3'] if row['profile_feature_3'] else "",
                    'profile_feature_4': row['profile_feature_4'] if row['profile_feature_4'] else "",
                    'profile_feature_5': row['profile_feature_5'] if row['profile_feature_5'] else "",
                    'profile_feature_6': row['profile_feature_6'] if row['profile_feature_6'] else ""
                }
    
    def _load_poi_info(self):
        """Load POI info data"""
        filename = self.input_files['poi_info']
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                poi_id = row['poi_id']
                self.poi_info[poi_id] = {
                    'poi_id': poi_id,
                    'normalized_score': row['normalized_score'],
                    'geographic_id': row['geographic_id'],
                    'category_id': row['category_id'],
                    'administrative_region_id': row['administrative_region_id']
                }
    
    def _load_geographic_poi_info(self):
        """Load POI info aggregated by geographic ID"""
        filename = self.input_files['poi_info_groupby_geographic']
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                geographic_id = row['geographic_id']
                self.geographic_poi_info[geographic_id] = row['poi_info']
    
    def get_poi_neg_sample_ids(self, target_poi_index, min_index, max_index, sample_num, label_poi_index=''):
        """Random negative sampling function"""
        sampled = set()
        result = []
        if max_index < sample_num + 1:
            return result
        while len(sampled) < sample_num:
            num = random.randint(min_index, max_index)
            if num not in sampled and str(num) != str(target_poi_index) and str(num) != str(label_poi_index):
                sampled.add(num)
                result.append(num)
        return result
    
    def neg_sampling_within_geographic(self, target_poi_index, poi_geographic_15_index, geographic_poi_info, sample_num):
        """Negative sampling within geographic"""
        result = []
        poi_info_list = geographic_poi_info.split(";")
        if sample_num >= len(poi_info_list) - 1:
            for i in range(len(poi_info_list)):
                cur_poi_info = poi_info_list[i]
                cur_poi_info_list = cur_poi_info.split(",")
                cur_poi_id = cur_poi_info_list[0]
                cur_poi_base_score = cur_poi_info_list[1]
                cur_poi_category_id = cur_poi_info_list[2]
                cur_poi_adcode = cur_poi_info_list[3]
                if int(cur_poi_id) == int(target_poi_index):
                    continue 
                feature_list = [cur_poi_id, poi_geographic_15_index, cur_poi_base_score, cur_poi_category_id, cur_poi_adcode]
                result.append(",".join(feature_list))
        else:
            sampled = set()
            count = 0
            while len(sampled) < sample_num:
                num = random.randint(0, len(poi_info_list) - 1)
                if num not in sampled:
                    cur_poi_info = poi_info_list[num]
                    cur_poi_info_list = cur_poi_info.split(",")
                    cur_poi_id = cur_poi_info_list[0]
                    cur_poi_base_score = cur_poi_info_list[1]
                    cur_poi_category_id = cur_poi_info_list[2]
                    cur_poi_adcode = cur_poi_info_list[3]
                    if int(cur_poi_id) == int(target_poi_index):
                        continue 
                    feature_list = [cur_poi_id, poi_geographic_15_index, cur_poi_base_score, cur_poi_category_id, cur_poi_adcode]
                    result.append(",".join(feature_list))
                    sampled.add(num)
                    count = count + 1
        return ";".join(result)
    
    def process_user_data(self, user_id):
        """Process all data for a single user"""
        if user_id not in self.user_actions:
            return None
            
        user_actions = self.user_actions[user_id]
        processed_actions = []
        
        # Process each action
        for action in user_actions:
            # Associate target POI info
            poi_id = action['poi_id']
            action['target_poi_normalized_score'] = ''
            action['target_poi_geographic_id'] = ''
            action['target_poi_category_id'] = ''
            action['target_poi_administrative_region_id'] = ''
            if poi_id in self.poi_info:
                poi_data = self.poi_info[poi_id]
                action['target_poi_normalized_score'] = poi_data['normalized_score']
                action['target_poi_geographic_id'] = poi_data['geographic_id']
                action['target_poi_category_id'] = poi_data['category_id']
                action['target_poi_administrative_region_id'] = poi_data['administrative_region_id']
            
            # Process via point info
            via_info = ""
            if action['via_poi_id'] and action['via_poi_id'] != "":
                via_poi_id = action['via_poi_id']
                if via_poi_id in self.poi_info:
                    via_poi_data = self.poi_info[via_poi_id]
                    via_info = ",".join([
                        via_poi_id,
                        via_poi_data['geographic_id'],
                        via_poi_data['normalized_score'],
                        via_poi_data['category_id'],
                        via_poi_data['administrative_region_id']
                    ])
            action['via_info'] = via_info
            
            # Random negative sampling
            label_poi_id = action['via_poi_id'] if action['via_poi_id'] else poi_id
            neg_sample_ids = self.get_poi_neg_sample_ids(
                label_poi_id, 0, self.max_poi_index, self.neg_sample_num
            )
            
            # Construct negative sample info
            negative_samples = []
            for neg_id in neg_sample_ids:
                neg_id_str = str(neg_id)
                if neg_id_str in self.poi_info:
                    neg_poi = self.poi_info[neg_id_str]
                    neg_sample_info = ",".join([
                        neg_id_str,
                        neg_poi['geographic_id'],
                        neg_poi['normalized_score'],
                        neg_poi['category_id'],
                        neg_poi['administrative_region_id']
                    ])
                    negative_samples.append(neg_sample_info)
            action['negative_samples'] = ";".join(negative_samples)
            


            # Negative sampling within Geographic
            if action['via_poi_id'] and action['via_poi_id'] in self.poi_info:
                neg_sample_geographic_index = self.poi_info[action['via_poi_id']]['geographic_id']
            else:
                neg_sample_geographic_index = action['target_poi_geographic_id']
            if neg_sample_geographic_index in self.geographic_poi_info:
                geographic_negative_samples = self.neg_sampling_within_geographic(
                    label_poi_id,
                    neg_sample_geographic_index,
                    self.geographic_poi_info[neg_sample_geographic_index],
                    self.tile_sample_num
                )
                action['geographic_negative_samples'] = geographic_negative_samples
            else:
                action['geographic_negative_samples'] = ""
            
            processed_actions.append(action)
        
        return processed_actions
    
    def multi_task_label_infer(self, seq_list, uid='', cut_num=50):
        """Multi-task label inference"""
        # Truncate to recent sequence
        seq_list = seq_list[-cut_num:]
        
        # First divide into sessions, then take last action as intention truth
        seq_list = self._poi_infer_real_value(seq_list, uid)
        
        return '&'.join(map(str, seq_list[-cut_num:]))
    
    def _poi_infer_real_value(self, seq_list, uid=''):
        """POI real value inference"""
        priority_dict = {
            "1": 0,  
            "0": 1,  
            "7": 2,  
            "2": 2,  
            "3": 3,  
            "4": 4,  
            "6": 5,  
            "5": 6   
        }
        
        len_seq_list = len(seq_list)
        seq_action_list = []
        tmp_session_list = []
        infer_res_list = []
        final_out_list = []
        
        session_count = 0
        
        for i in range(len_seq_list):
            cur_action_list = seq_list[i].split('|')
            seq_action_list.append(cur_action_list)
            timestamp_unix = cur_action_list[0]
            action_type = cur_action_list[1]
            cur_action_type_priority = priority_dict[action_type]
            poi_index = cur_action_list[3]
            via_info = cur_action_list[11] if len(cur_action_list) > 11 else ""
            tmp_session_list.append([cur_action_type_priority, int(timestamp_unix), cur_action_list])
            
            # Check if current session should end
            should_end_session = False
            if i < len_seq_list - 1:
                next_action_list = seq_list[i+1].split('|')
                next_timestamp_unix = next_action_list[0]
                next_poi_index = next_action_list[3]
                next_via_info = next_action_list[11] if len(next_action_list) > 11 else ""
                time_diff = int(next_timestamp_unix) - int(timestamp_unix)
                merge_condition = (poi_index == next_poi_index and 
                                 via_info == "" and 
                                 time_diff <= 5 * 60 * 1000)
                
                # Check session merge condition
                
                # If next action doesn't meet merge condition with current action, end session
                if not merge_condition:
                    should_end_session = True
            else:
                # Last action, must end session
                should_end_session = True
                    
            if should_end_session and len(tmp_session_list) > 0:
                session_count += 1
                one_action_list_infer = self._infer_action(tmp_session_list)
                infer_res_list.append(one_action_list_infer)
                tmp_session_list = []
        
        # Session merge completed
        
        # Add depart time truth for actions without found truth
        len_infer_res_list = len(infer_res_list)
        for i in range(len_infer_res_list):
            cur_infer_res = infer_res_list[i]
            if len(cur_infer_res) == 15:  # Departure time truth already found
                pass
            else:  # Search for truth in future
                future_travel_time_gap = self._find_future_travel_time_gap(cur_infer_res, seq_action_list)
                cur_infer_res.append(future_travel_time_gap)
                infer_res_list[i] = cur_infer_res
        
        for info in infer_res_list:
            final_out_list.append('|'.join(map(str, info)))
        
        return final_out_list
    
    def _infer_action(self, tmp_session_list):
        """Infer action for a single session"""
        # Sort by type priority and time
        tmp_session_list_sorted = sorted(tmp_session_list, key=lambda x: (x[0], -x[1]), reverse=True)
        first_priority_poi_action_type = tmp_session_list_sorted[0][2][1]
        
        len_tmp_session_list = len(tmp_session_list)
        first_route_data = []
        
        # Find first route action
        for i in range(len_tmp_session_list):
            if tmp_session_list[i][0] == 2:  # route
                first_route_data = tmp_session_list[i][2]
                break
                
        if first_route_data == []:  # No route action
            first_route_data = tmp_session_list_sorted[0][2]
        
        # Find navigation action
        navi_data = []
        for i in range(len_tmp_session_list):
            if int(tmp_session_list[i][2][0]) < int(first_route_data[0]):
                continue
            if tmp_session_list[i][0] == 3:  # navi
                navi_data = tmp_session_list[i][2]
                break
        
        data_base = copy.deepcopy(first_route_data)
        data_base[1] = first_priority_poi_action_type
        
        # Attach travel mode truth
        travel_mode_label = ""
        for i in range(len_tmp_session_list-1, -1, -1):
            if int(tmp_session_list[i][2][0]) < int(first_route_data[0]):
                break
            if tmp_session_list[i][2][10] == "":
                continue
            else:
                travel_mode_label = tmp_session_list[i][2][10]
                break
        
        if travel_mode_label != "":
            data_base[10] = travel_mode_label
        
        # Attach departure time truth
        if navi_data != []:
            future_travel_time_gap = int(navi_data[0]) - int(data_base[0])
            data_base.append(future_travel_time_gap)
        
        return data_base
    
    def _find_future_travel_time_gap(self, cur_action_list, seq_action_list):
        """Find departure time gap"""
        cur_timestamp_unix = cur_action_list[0]
        cur_geographic_id = cur_action_list[2]
        cur_poi_index = cur_action_list[3]
        len_seq_list = len(seq_action_list)
        
        for i in range(len_seq_list):
            next_action_list = seq_action_list[i]
            next_timestamp_unix = next_action_list[0]
            next_action_type = next_action_list[1]
            
            if next_action_type != '3' or int(next_timestamp_unix) < int(cur_timestamp_unix):
                continue
                
            time_gap = int(next_timestamp_unix) - int(cur_timestamp_unix)
            if time_gap > 2 * 24 * 60 * 60 * 1000:  # Over 2 days
                return -1
                
            next_geographic_id = next_action_list[2]
            next_poi_index = next_action_list[3]
            
            if (next_action_type == '3' and 
                next_geographic_id == cur_geographic_id and 
                next_poi_index == cur_poi_index):
                return time_gap
        
        return -1
    
    def process_all_users(self):
        """Process all user data"""
        print("Starting to process user data...")
        results = []
        
        for user_id in self.user_actions.keys():
            # Process user action data
            processed_actions = self.process_user_data(user_id)
            if not processed_actions:
                continue
            
            # Construct action info string
            action_strings = []
            action_times = []  # For debugging and validation of sorting
            for action in processed_actions:
                action_times.append(action['timestamp'])  # Record timestamp for validation
                # Ensure all fields are string type
                action_info = '|'.join([
                    str(action['timestamp']),
                    str(action['action_type']) if action['action_type'] is not None else '',
                    str(action['geographic_id']) if action['geographic_id'] is not None else '',
                    str(action['poi_id']) if action['poi_id'] is not None else '',
                    str(action['target_poi_geographic_id']) if action['target_poi_geographic_id'] is not None else '',
                    str(action['target_poi_normalized_score']) if action['target_poi_normalized_score'] is not None else '',
                    str(action['target_poi_category_id']) if action['target_poi_category_id'] is not None else '',
                    str(action['target_poi_administrative_region_id']) if action['target_poi_administrative_region_id'] is not None else '',
                    str(action['administrative_region_id']) if action['administrative_region_id'] is not None else '',
                    str(action['weather']) if action['weather'] is not None else '',
                    str(action['travel_mode']) if action['travel_mode'] is not None else '',
                    str(action['via_info']) if action['via_info'] is not None else '',
                    str(action['negative_samples']) if action['negative_samples'] is not None else '',
                    str(action['geographic_negative_samples']) if action['geographic_negative_samples'] is not None else ''
                ])
                action_strings.append(action_info)
            
            # Validate sorting correctness
            if action_times != sorted(action_times):
                print("Warning: action sorting may be incorrect!")
                print("Original order:", action_times)
                print("After sorting:", sorted(action_times))
            
            # Concatenate sequence info
            seq_info = '&'.join(action_strings)
            
            # Multi-task label inference - split seq_info by & into individual actions list
            action_list = seq_info.split('&')
            seq_info_with_labels = self.multi_task_label_infer(action_list, user_id, self.max_seq_length)
            
            # Get user profile
            profile = self.user_profiles.get(user_id, {})
            profile_feature_1 = profile.get('profile_feature_1', '')
            profile_feature_2 = profile.get('profile_feature_2', '')
            profile_feature_3 = profile.get('profile_feature_3', '')
            profile_feature_4 = profile.get('profile_feature_4', '')
            profile_feature_5 = profile.get('profile_feature_5', '')
            profile_feature_6 = profile.get('profile_feature_6', '')
            
            # Construct training data
            train_data = {
                'seq_info': seq_info_with_labels,
                'profile_feature_1': profile_feature_1,
                'profile_feature_2': profile_feature_2,
                'profile_feature_3': profile_feature_3,
                'profile_feature_4': profile_feature_4,
                'profile_feature_5': profile_feature_5,
                'profile_feature_6': profile_feature_6
            }
            
            # Convert to JSON format
            seq_info_json = json.dumps({"content": train_data})
            
            # rand_1 is a random number for test/valid dataset splitting 
            # as the total number of users is large
            results.append({
                'user_id': user_id,
                'seq_info': seq_info_json,
                'rand_1': random.random()
            })
            
            if len(results) % 100 == 0:
                print("Processed {} users".format(len(results)))
        
        print("User data processing completed, total {} users processed".format(len(results)))
        return results
    
    def save_results(self, results, output_filename=None):
        """Save results to file"""
        if output_filename is None:
            output_filename = 'multi_task_input_seq_all.csv'
        output_file = os.path.join(self.output_dir, output_filename)
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Created output directory: {}".format(self.output_dir))
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("user_id\tseq_info\trand_1\n")
            
            # Write data
            for result in results:
                line = "{}\t{}\t{}\n".format(
                    result['user_id'],
                    result['seq_info'],
                    result['rand_1']
                )
                f.write(line)
        
        print("Results saved to: {}".format(output_file))
        return output_file

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-task learning data processing program')
    parser.add_argument('--input-dir', '-i', default='./raw_data', 
                       help='Input data directory (default: ./raw_data)')
    parser.add_argument('--output-dir', '-o', default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--output-file', '-f', default='multi_task_input_seq_all.csv',
                       help='Output file name (default: multi_task_input_seq_all.csv)')
    
    # Four input file arguments
    parser.add_argument('--user-action-file', default='user_action.csv',
                       help='User action data file (default: user_action.csv)')
    parser.add_argument('--user-profile-file', default='user_profile.csv',
                       help='User profile data file (default: user_profile.csv)')
    parser.add_argument('--poi-info-file', default='poi_info.csv',
                       help='POI info data file (default: poi_info.csv)')
    parser.add_argument('--poi-geographic-file', default='poi_info_groupby_geographic.csv',
                       help='POI data file (grouped by geographic ID) (default: poi_info_groupby_geographic.csv)')
    
    # Other options
    parser.add_argument('--skip-column-check', action='store_true',
                       help='Skip column name check')
    parser.add_argument('--max-actions', type=int, default=None,
                       help='Only read the first N interaction rows (default: all)')
    
    args = parser.parse_args()
    
    # Build input file configuration
    input_files = {
        'user_action': args.user_action_file,
        'user_profile': args.user_profile_file,
        'poi_info': args.poi_info_file,
        'poi_info_groupby_geographic': args.poi_geographic_file
    }
    
    print("=" * 60)
    print("Multi-task learning data processing program")
    print("=" * 60)
    print("Input directory: {}".format(args.input_dir))
    print("Output directory: {}".format(args.output_dir))
    print("Output file: {}".format(args.output_file))
    print("\nInput file configuration:")
    for table_name, filename in input_files.items():
        print("  {}: {}".format(table_name, filename))
    print("=" * 60)
    
    # Initialize processor
    processor = DataProcessor(args.input_dir, args.output_dir, input_files=input_files,
                              max_actions=args.max_actions)
    
    # Load data (including column check)
    if not processor.load_data(check_columns=not args.skip_column_check):
        print("Data loading failed, program exit")
        sys.exit(1)
    
    # Process all user data
    results = processor.process_all_users()
    
    # Save results
    output_file = processor.save_results(results, args.output_file)
    
    print("=" * 60)
    print("Data processing completed!")
    print("Output file: {}".format(output_file))
    print("=" * 60)

if __name__ == '__main__':
    main()