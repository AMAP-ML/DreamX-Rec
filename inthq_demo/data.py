import math

import torch
from torch.utils.data import Dataset

from data_process import DataProcessor, FeaturePostProcessor, map_dict, mfc

try:
    from .config import TASKS, TASK_TOKEN_OFFSET, POI_TASKS
except ImportError:
    from config import TASKS, TASK_TOKEN_OFFSET, POI_TASKS

TOK_PER_SESSION = mfc.action_cnt_in_session
N_PROFILE = len(mfc.u_feature_name_total)
N_NEG = mfc.negative_sample_num
PAD_POS = 1 << 20

CTX_INT_MAP = {
    "type_detail": "type_index_detail",
    "geo": "action_list_exp_geographic",
    "user_adcode": "action_list_exp_user_loc_administrative_region_index",
    "weather": "action_list_exp_source_condition_index",
    "poi_id": "action_list_poi_id",
    "poi_geo": "action_list_poi_geographic",
    "poi_cat": "action_list_poi_category",
    "poi_adcode": "action_list_poi_administrative_region",
    "travel_mode": "action_list_travel_mode",
    "profile_id": "u_feature_id",
}
POI_LABEL_MAP = {
    "id": "label_poi_id",
    "geo": "label_poi_geographic",
    "cat": "label_poi_category",
    "adcode": "label_poi_administrative_region",
}
POI_NEG_MAP = {
    "id": "label_neg_poi_id",
    "geo": "label_neg_poi_geographic",
    "cat": "label_neg_poi_category",
    "adcode": "label_neg_poi_administrative_region",
}
MODEL_TYPE_INDEX = {
    "scenario": 0,
    "intention": 2,
    "feedback": 3,
    "user_profile": 4,
}
PUBLIC_TO_MODEL_TYPE = {
    map_dict.seq_type_dict[name]: model_index
    for name, model_index in MODEL_TYPE_INDEX.items()
}


def _reorder_actions(seq, n_act):
    ordered = []
    for start in range(0, n_act, TOK_PER_SESSION):
        feedback, intention, scenario = seq[start:start + TOK_PER_SESSION]
        ordered.extend((scenario, intention, feedback))
    return ordered


def _rows(flat, n_tok, width):
    return [flat[i * width:(i + 1) * width] for i in range(n_tok)]


def _pad(seq, size, fill):
    return list(seq) + [fill] * (size - len(seq))


def _time_bucket(ts, ref):
    if ts is None or ts < 0:
        return 0
    return 1 + min(30, int(math.log2(1.0 + max(0, ref - ts) / 60.0)))


def build_raw_features(raw_dir, max_actions):
    processor = DataProcessor(str(raw_dir), max_actions=max_actions)
    processor.load_data(check_columns=False)
    rows = processor.process_all_users()
    post = FeaturePostProcessor()
    features = []
    for row in rows:
        feature = post.process_single_sample(row)
        if feature:
            features.append(feature)
    return features


def encode(feature, max_sessions):
    n_tok = len(feature["type_index"])
    n_act = n_tok - N_PROFILE
    n_sess = n_act // TOK_PER_SESSION
    if n_sess > max_sessions:
        raise ValueError("sequence longer than max_sessions")
    n_action_slots = TOK_PER_SESSION * max_sessions
    n_ctx = n_action_slots + N_PROFILE

    def context_values(key, fill):
        actions = _pad(_reorder_actions(feature[key], n_act), n_action_slots, fill)
        return actions + list(feature[key][n_act:n_act + N_PROFILE])

    ctx = {}
    ordered_type = context_values("type_index", -1)
    remapped_type = [PUBLIC_TO_MODEL_TYPE[value] if value >= 0 else -1 for value in ordered_type]
    ctx["type_index"] = torch.tensor(remapped_type, dtype=torch.long)
    for key, src in CTX_INT_MAP.items():
        ctx[key] = torch.tensor(context_values(src, -1), dtype=torch.long)
    ctx["poi_score"] = torch.tensor(
        context_values("action_list_poi_base_score", 0.0), dtype=torch.float
    )

    times = context_values("action_list_time_feature", -1)
    ref = max([t for t in times if t >= 0] or [0])
    ctx["time_bucket"] = torch.tensor(
        [_time_bucket(t, ref) for t in times], dtype=torch.long
    )
    ctx["time"] = torch.tensor(times, dtype=torch.long)

    session_rank = torch.arange(max_sessions, dtype=torch.long).repeat_interleave(TOK_PER_SESSION)
    local = torch.tensor([0, 2, 3], dtype=torch.long).repeat(max_sessions)
    profile_rank = torch.arange(N_PROFILE, dtype=torch.long)
    ctx["cti"] = torch.cat((-4 * session_rank + local, -4 * max_sessions - profile_rank))
    ctx["gpos"] = torch.cat((4 * session_rank + local, 4 * max_sessions + profile_rank))
    ctx["valid"] = torch.tensor(
        [True] * n_act + [False] * (n_action_slots - n_act) + [True] * N_PROFILE,
        dtype=torch.bool,
    )

    pos_label = {k: _reorder_actions(feature[src], n_act) for k, src in POI_LABEL_MAP.items()}
    pos_label["score"] = _reorder_actions(feature["label_poi_score"], n_act)
    neg_label = {
        k: _reorder_actions(_rows(feature[src], n_tok, N_NEG), n_act)
        for k, src in POI_NEG_MAP.items()
    }
    neg_label["score"] = _reorder_actions(
        _rows(feature["label_neg_poi_score"], n_tok, N_NEG), n_act
    )
    mode_label = _reorder_actions(feature["label_travel_mode"], n_act)
    when_label = _reorder_actions(feature["label_future_travel"], n_act)

    task_cti, task_gpos, task_valid, task_time = [], [], [], []
    for session in range(max_sessions):
        for name in TASKS:
            live = session < n_sess
            task_local = TASK_TOKEN_OFFSET[name] + 1
            task_cti.append(-4 * session + task_local)
            task_gpos.append(4 * session + task_local)
            task_valid.append(live)
            source = TOK_PER_SESSION * session + TASK_TOKEN_OFFSET[name]
            task_time.append(times[source] if live else -1)
    task = {
        "cti": torch.tensor(task_cti, dtype=torch.long),
        "gpos": torch.tensor(task_gpos, dtype=torch.long),
        "valid": torch.tensor(task_valid, dtype=torch.bool),
        "time": torch.tensor(task_time, dtype=torch.long),
    }

    def host(session, name):
        return TOK_PER_SESSION * session + TASK_TOKEN_OFFSET[name]

    labels = {}
    for name in POI_TASKS:
        entry = {}
        for key in ("id", "geo", "cat", "adcode"):
            values = [pos_label[key][host(s, name)] if s < n_sess else -1 for s in range(max_sessions)]
            entry["pos_" + key] = torch.tensor(values, dtype=torch.long)
            rows = [
                neg_label[key][host(s, name)] if s < n_sess else [-1] * N_NEG
                for s in range(max_sessions)
            ]
            entry["neg_" + key] = torch.tensor(rows, dtype=torch.long)
        entry["pos_score"] = torch.tensor(
            [pos_label["score"][host(s, name)] if s < n_sess else 0.0 for s in range(max_sessions)],
            dtype=torch.float,
        )
        entry["neg_score"] = torch.tensor(
            [neg_label["score"][host(s, name)] if s < n_sess else [0.0] * N_NEG for s in range(max_sessions)],
            dtype=torch.float,
        )
        labels[name] = entry
    labels["how"] = torch.tensor(
        [mode_label[host(s, "how")] if s < n_sess else -1 for s in range(max_sessions)], dtype=torch.long
    )
    labels["when"] = torch.tensor(
        [when_label[host(s, "when")] if s < n_sess else -1 for s in range(max_sessions)], dtype=torch.long
    )

    holdout = torch.zeros(max_sessions, dtype=torch.bool)
    if n_sess:
        holdout[0] = True
    return {
        "ctx": ctx,
        "task": task,
        "labels": labels,
        "holdout": holdout,
        "n_sessions": n_sess,
    }


class IntTravelDemoDataset(Dataset):
    def __init__(self, raw_dir, max_actions, max_sessions):
        features = build_raw_features(raw_dir, max_actions)
        self.user_ids = [f["user_id"] for f in features]
        self.samples = [encode(f, max_sessions) for f in features]
        self.n_sessions = [s["n_sessions"] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "ctx": sample["ctx"],
            "task": sample["task"],
            "labels": sample["labels"],
            "holdout": sample["holdout"],
        }


def move_to(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    return {k: move_to(v, device) for k, v in batch.items()}
