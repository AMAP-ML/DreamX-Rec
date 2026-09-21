import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from .config import TASKS, TASK_TOKEN_OFFSET, POI_TASKS
except ImportError:
    from config import TASKS, TASK_TOKEN_OFFSET, POI_TASKS

TOK_PER_SESSION = 3
N_PROFILE = 6
N_NEG = 64
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
PUBLIC_TO_MODEL_TYPE = {0: 0, 1: 2, 2: 3, 3: 4}
NEGATIVE_FIELDS = set(POI_NEG_MAP.values()) | {"label_neg_poi_score"}
FEATURE_FIELDS = {
    "type_index",
    "action_list_time_feature",
    "action_list_poi_base_score",
    "label_poi_score",
    "label_travel_mode",
    "label_future_travel",
    *CTX_INT_MAP.values(),
    *POI_LABEL_MAP.values(),
    *NEGATIVE_FIELDS,
}
REQUIRED_COLUMNS = FEATURE_FIELDS | {"user_id"}


def load_processed_features(processed_file, max_sessions):
    path = Path(processed_file)
    if not path.is_file():
        raise FileNotFoundError(f"processed feature data not found: {path}")
    features = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if not row["user_id"]:
                raise ValueError(f"{path}:{row_number} has invalid user_id")
            feature = {"user_id": row["user_id"], "rand_1": None}
            if "rand_1" in row and row["rand_1"]:
                try:
                    feature["rand_1"] = float(row["rand_1"])
                except ValueError as error:
                    raise ValueError(f"{path}:{row_number} has invalid rand_1") from error
                if not 0.0 <= feature["rand_1"] < 1.0:
                    raise ValueError(f"{path}:{row_number} has invalid rand_1")
            for name in FEATURE_FIELDS:
                try:
                    value = json.loads(row[name])
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{row_number} has invalid {name}") from error
                if not isinstance(value, list):
                    raise ValueError(f"{path}:{row_number} field {name} is not an array")
                feature[name] = value
            _validate_feature(feature, path, row_number, max_sessions)
            features.append(feature)
    return features


def _validate_feature(feature, path, row_number, max_sessions):
    n_tok = len(feature["type_index"])
    n_act = n_tok - N_PROFILE
    if n_act <= 0 or n_act % TOK_PER_SESSION:
        raise ValueError(f"{path}:{row_number} has invalid token count {n_tok}")
    if n_act // TOK_PER_SESSION > max_sessions:
        raise ValueError(f"{path}:{row_number} exceeds max_sessions={max_sessions}")
    for name in FEATURE_FIELDS - NEGATIVE_FIELDS:
        if len(feature[name]) != n_tok:
            raise ValueError(f"{path}:{row_number} field {name} has invalid length")
    for name in NEGATIVE_FIELDS:
        if len(feature[name]) != n_tok * N_NEG:
            raise ValueError(f"{path}:{row_number} field {name} has invalid length")
    for start in range(0, n_act, TOK_PER_SESSION):
        if feature["type_index"][start:start + TOK_PER_SESSION] != [2, 1, 0]:
            raise ValueError(f"{path}:{row_number} has invalid [F,I,S] layout")
    if feature["type_index"][n_act:] != [3] * N_PROFILE:
        raise ValueError(f"{path}:{row_number} has invalid profile layout")


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


def encode(feature, max_sessions):
    n_tok = len(feature["type_index"])
    n_act = n_tok - N_PROFILE
    n_sess = n_act // TOK_PER_SESSION
    if n_sess > max_sessions:
        raise ValueError("sequence longer than max_sessions")
    n_action_slots = TOK_PER_SESSION * max_sessions
    n_ctx = n_action_slots + N_PROFILE

    def context_values(key, fill):
        actions = _reorder_actions(feature[key], n_act)
        profile = list(feature[key][n_act:n_act + N_PROFILE])
        return _pad(actions + profile, n_ctx, fill)

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
    action_cti = -4 * session_rank + local
    action_gpos = 4 * session_rank + local
    profile_rank = torch.arange(N_PROFILE, dtype=torch.long)
    profile_cti = -4 * max_sessions - profile_rank
    profile_gpos = 4 * max_sessions + profile_rank
    ctx["cti"] = torch.cat((action_cti[:n_act], profile_cti, action_cti[n_act:]))
    ctx["gpos"] = torch.cat((action_gpos[:n_act], profile_gpos, action_gpos[n_act:]))
    ctx["valid"] = torch.tensor(
        [True] * (n_act + N_PROFILE) + [False] * (n_action_slots - n_act),
        dtype=torch.bool,
    )
    ctx["profile_positions"] = torch.arange(n_act, n_act + N_PROFILE, dtype=torch.long)

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

    return {
        "ctx": ctx,
        "task": task,
        "labels": labels,
        "n_sessions": n_sess,
    }


class IntTravelDemoDataset(Dataset):
    def __init__(self, processed_file, max_sessions):
        features = load_processed_features(processed_file, max_sessions)
        self.user_ids = [f["user_id"] for f in features]
        self.rand_values = [f["rand_1"] for f in features]
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
        }


def move_to(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    return {k: move_to(v, device) for k, v in batch.items()}
