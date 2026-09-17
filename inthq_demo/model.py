import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import N_PROFILE, POI_TASKS, TASKS
except ImportError:
    from config import N_PROFILE, POI_TASKS, TASKS

CLS_TYPE_INDEX = 1


def recis_available():
    """True when RecIS can be imported (it is an optional dependency)."""
    try:
        import recis  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_id_backend(name):
    """Resolve cfg.embedding_backend to 'torch' or 'recis'."""
    if name not in ("auto", "torch", "recis"):
        raise ValueError("embedding_backend must be 'auto', 'torch' or 'recis', got %r" % name)
    if name == "auto":
        return "recis" if recis_available() else "torch"
    if name == "recis" and not recis_available():
        raise RuntimeError(
            "embedding_backend='recis' but RecIS is not importable; see requirements.txt")
    return name


def _init_linear(linear):
    std = 2.0 / (linear.in_features ** 0.5)
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class MultiHashEmbedding(nn.Module):

    def __init__(self, primes, sub_dim, sparse=False):
        super().__init__()
        self.primes = tuple(primes)
        self.tables = nn.ModuleList(
            nn.Embedding(p, sub_dim, sparse=sparse) for p in self.primes)

    def forward(self, ids):
        safe = ids.clamp(min=0)
        parts = [table(torch.remainder(safe, prime))
                 for prime, table in zip(self.primes, self.tables)]
        out = torch.cat(parts, dim=-1)
        return out * (ids >= 0).unsqueeze(-1).to(out.dtype)


class RecisMultiHashEmbedding(nn.Module):
    """Multi-hash id embedding backed by RecIS dynamic hash tables.

    The table grows on demand, so there is no fixed vocabulary and no reserved padding row;
    ids < 0 are zeroed on the way out. `recis` is imported lazily so the pure-PyTorch path
    never needs it installed.
    """

    def __init__(self, primes, sub_dim, init_std, name_prefix, block_size=10240):
        super().__init__()
        from recis.nn import DynamicEmbedding, EmbeddingOption
        from recis.nn.initializers import TruncNormalInitializer

        # The hash-table kernels are CUDA-only, so fail fast instead of feeding host pointers.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The RecIS embedding backend requires CUDA; use embedding_backend='torch'.")

        self.primes = tuple(primes)
        self.sub_dim = sub_dim
        self.tables = nn.ModuleList(
            DynamicEmbedding(EmbeddingOption(
                embedding_dim=sub_dim,
                block_size=block_size,
                dtype=torch.float32,
                device=torch.device("cuda"),
                trainable=True,
                shared_name="%s_%d" % (name_prefix, i + 1),
                coalesced=True,
                initializer=TruncNormalInitializer(std=init_std),
                # One id per row, so summing a single element is the identity. This keeps the
                # module shape-agnostic: the same tables are read at several different shapes.
                combiner="sum",
            ))
            for i in range(len(self.primes))
        )

    def forward(self, ids):
        flat = ids.reshape(-1, 1).long()
        if flat.device.type != "cuda":
            flat = flat.cuda()
        safe = flat.clamp(min=0)
        parts = [table(torch.remainder(safe, prime))
                 for prime, table in zip(self.primes, self.tables)]
        out = torch.cat(parts, dim=-1).reshape(*ids.shape, len(self.primes) * self.sub_dim)
        return out * (ids >= 0).unsqueeze(-1).to(out.dtype).to(out.device)


class PoiEncoder(nn.Module):

    def __init__(self, cfg, poi_id, geo, category, adcode):
        super().__init__()
        self.poi_id, self.geo, self.category, self.adcode = poi_id, geo, category, adcode
        self.score = _init_linear(nn.Linear(1, cfg.d_model))
        self.score_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, poi_id, geo, category, adcode, score):
        mask = (poi_id != -1).unsqueeze(-1).to(self.score.weight.dtype)
        score_emb = F.relu(self.score_norm(self.score(score.unsqueeze(-1)))) * mask
        h = self.poi_id(poi_id) + self.geo(geo) + score_emb
        h = h + self.category(category) + self.adcode(adcode)
        return h * mask


class TokenEmbedding(nn.Module):

    def __init__(self, cfg, n_ctx, n_sessions):
        super().__init__()
        sp = cfg.sparse_embedding
        # common (all tokens)
        self.token_type = IDEmbeddingLayer(cfg.seq_type_vocab, cfg.d_model)
        self.detail = IDEmbeddingLayer(cfg.action_type_vocab, cfg.d_model)
        # S token: weather goes through a 48 -> d_model MLP; tile/adcode are added directly
        self.weather = IDEmbeddingLayer(cfg.weather_vocab, cfg.weather_dim)
        self.weather_proj = _init_linear(nn.Linear(cfg.weather_dim, cfg.d_model))
        self.weather_norm = nn.LayerNorm(cfg.d_model)
        # F token (also the `how` classifier table)
        self.travel_mode = IDEmbeddingLayer(cfg.n_travel_mode, cfg.d_model)
        # U token
        self.profile = IDEmbeddingLayer(cfg.profile_vocab, cfg.d_model)
        backend = _resolve_id_backend(cfg.embedding_backend)
        self.id_backend = backend
        if backend == "recis":
            self.poi_id = RecisMultiHashEmbedding(
                cfg.hash_primes, cfg.sub_dim, cfg.emb_init_std, "global_poi_emb_table")
            self.geo = RecisMultiHashEmbedding(
                cfg.hash_primes, cfg.sub_dim, cfg.emb_init_std, "global_tile_emb_table")
        else:
            self.poi_id = MultiHashEmbedding(cfg.hash_primes, cfg.sub_dim, sparse=sp)
            self.geo = MultiHashEmbedding(cfg.hash_primes, cfg.sub_dim, sparse=sp)
        self.adcode = IDEmbeddingLayer(cfg.adcode_vocab, cfg.d_model)
        self.category = IDEmbeddingLayer(cfg.category_vocab, cfg.d_model)
        self.poi = PoiEncoder(cfg, self.poi_id, self.geo, self.category, self.adcode)
        self.n_ctx = n_ctx
        self._init_embeddings(cfg.emb_init_std)

    def _init_embeddings(self, std):
        for holder in (self.poi_id, self.geo):
            for m in holder.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.trunc_normal_(m.weight, std=std)

    def forward(self, ctx):
        dtype = self.weather_proj.weight.dtype
        valid = ctx["valid"].unsqueeze(-1).to(dtype)
        type_index = ctx["type_index"]
        is_s = (type_index == 0).unsqueeze(-1).to(dtype)
        is_i = (type_index == 2).unsqueeze(-1).to(dtype)
        is_f = (type_index == 3).unsqueeze(-1).to(dtype)
        is_u = (type_index == 4).unsqueeze(-1).to(dtype)

        weather = F.relu(self.weather_norm(self.weather_proj(self.weather(ctx["weather"])))) * is_s
        emb_s = (weather + self.geo(ctx["geo"]) + self.adcode(ctx["user_adcode"])) * is_s
        emb_i = self.poi(ctx["poi_id"], ctx["poi_geo"], ctx["poi_cat"],
                         ctx["poi_adcode"], ctx["poi_score"]) * is_i
        emb_f = self.travel_mode(ctx["travel_mode"]) * is_f
        emb_u = self.profile(ctx["profile_id"]) * is_u
        common = self.token_type(type_index) + self.detail(ctx["type_detail"])
        return (emb_s + emb_i + emb_f + emb_u + common) * valid


class HstuTimeDeltaCross(nn.Module):

    def __init__(self, time_buckets, alibi_lambda):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.tensor(float(alibi_lambda)))
        self.time_buckets = time_buckets

    def forward(self, q_time, k_time, allow):
        delta = (q_time.unsqueeze(2) - k_time.unsqueeze(1)).abs().clamp(min=1).float()
        bucket = (torch.log(delta) / 0.301).clamp(0, self.time_buckets).detach()
        return -self.lambda_ * bucket * allow


class IntHQAttention(nn.Module):

    def __init__(self, dim, max_rel):
        super().__init__()
        self.q_proj = _init_linear(nn.Linear(dim, 2 * dim))
        self.kv_proj = _init_linear(nn.Linear(dim, 2 * dim))
        self.out = _init_linear(nn.Linear(dim, dim))
        self.av_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

        self.rel = nn.Parameter(torch.randn(2 * max_rel + 1) * 0.02)
        self.max_rel = max_rel

    def forward(self, q_in, kv_in, q_gpos, k_gpos, allow, time_delta, q_valid=None):
        qu = F.silu(self.q_proj(q_in))
        q, u = qu.chunk(2, dim=-1)
        kv = F.silu(self.kv_proj(kv_in))
        k, v = kv.chunk(2, dim=-1)

        sim = q @ k.transpose(-1, -2)
        rel_idx = (k_gpos.unsqueeze(1) - q_gpos.unsqueeze(2) + self.max_rel).clamp(
            0, 2 * self.max_rel
        )
        sim = sim + self.rel[rel_idx] + time_delta
        sim = F.silu(sim) * allow
        denom = sim.ne(0).sum(-1, keepdim=True).clamp(min=1).float()
        av = (sim / denom) @ v
        out = self.out_norm(self.out(self.av_norm(av) * u))

        if q_valid is not None:
            out = out * q_valid.unsqueeze(-1).to(out.dtype)
        return out


class DualStreamLayer(nn.Module):

    def __init__(self, cfg, max_rel, layer_idx):
        super().__init__()
        self.ctx_pre_norm = nn.LayerNorm(cfg.d_model)
        self.q_pre_norm = nn.LayerNorm(cfg.d_model)
        self.ctx_attn = IntHQAttention(cfg.d_model, max_rel)
        self.q_attn = IntHQAttention(cfg.d_model, max_rel)
        self.q_out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, h_ctx, h_task, co):
        ctx_valid, task_valid = co["ctx_valid"], co["task_valid"]
        norm_ctx = self.ctx_pre_norm(h_ctx)
        norm_task = self.q_pre_norm(h_task)

        sif_out = self.ctx_attn(
            norm_ctx, norm_ctx, co["ctx_gpos"], co["ctx_gpos"], co["allow_cc"],
            co["time_cc"], ctx_valid)
        q_cross = self.q_attn(
            norm_task, norm_ctx, co["task_gpos"], co["ctx_gpos"], co["allow_qc"],
            co["time_qc"], task_valid)
        q_self = self.q_attn(
            norm_task, norm_task, co["task_gpos"], co["task_gpos"], co["allow_qq"],
            co["time_qq"], task_valid)

        out_ctx = h_ctx + sif_out
        out_task = h_task + q_cross + q_self
        return out_ctx, out_task, self.q_out_norm(out_task)


class HierarchicalQuery(nn.Module):

    def __init__(self, cfg, n_layers, task_emb_size):
        super().__init__()
        attn_dim = cfg.hq_attn_dim
        heads = cfg.hq_num_heads
        assert attn_dim % heads == 0 and cfg.d_model % heads == 0
        self.n_layers = n_layers
        self.attn_dim = attn_dim
        self.n_heads = heads
        self.d_head = attn_dim // heads
        self.v_head = cfg.d_model // heads
        self.scale = self.d_head ** -0.5
        self.query_proj = nn.Linear(task_emb_size, attn_dim)
        self.key = nn.Linear(cfg.d_model, attn_dim)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)
        self.depth = nn.Parameter(torch.randn(n_layers, attn_dim) * 0.02)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, stack, task_ids, task_table):
        b, n_slots, n_layers, d = stack.shape
        k = (self.key(stack) + self.depth.view(1, 1, n_layers, self.attn_dim))
        v = self.value(stack)
        q = self.query_proj(task_table.weight)[task_ids]
        qh = q.view(1, n_slots, 1, self.n_heads, self.d_head)
        kh = k.view(b, n_slots, n_layers, self.n_heads, self.d_head)
        vh = v.view(b, n_slots, n_layers, self.n_heads, self.v_head)
        weight = (kh * qh).sum(-1).mul(self.scale).softmax(dim=2)
        pooled = torch.einsum("bnlh,bnlhd->bnhd", weight, vh).reshape(b, n_slots, d)
        last = stack[:, :, -1]
        return torch.cat([last, self.norm(last + pooled)], dim=-1)


class ScenarioFactorLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_experts):
        super().__init__()
        self.experts_w = nn.Parameter(torch.randn(n_experts, in_dim, out_dim) * (2.0 / math.sqrt(in_dim)))
        self.experts_b = nn.Parameter(torch.zeros(n_experts, out_dim))

    def mix(self, expert_weight, extra_w=None, extra_b=None):
        shared_count = self.experts_w.size(0)
        weight = torch.einsum(
            "be,eij->bij", expert_weight[:, :shared_count], self.experts_w
        )
        bias = torch.einsum(
            "be,ej->bj", expert_weight[:, :shared_count], self.experts_b
        )
        if extra_w is not None:
            private_weight = expert_weight[:, shared_count:]
            weight = weight + torch.einsum("be,eij->bij", private_weight, extra_w)
            bias = bias + torch.einsum("be,ej->bj", private_weight, extra_b)
        return weight, bias

    def apply_mixed(self, x, weight, bias, norm, mask=None):
        out = torch.einsum("bsi,bij->bsj", x, weight) + bias.unsqueeze(1)
        if mask is not None:
            out = out * mask
        out = norm(out)
        out = F.elu(out)
        if mask is not None:
            out = out * mask
        return out


class DSFNet(nn.Module):
    """Dynamic Scenario Factorization:
    scene embedding (task emb [+ profile]) -> layer1 shared MLP -> per-task layer2 ->
    per-expert weights (sigmoid * 2) mixing shared + task-private expert matrices."""

    SCENE_HIDDEN = 16

    def __init__(self, cfg, in_dim, n_tasks, task_table):
        super().__init__()
        self.n_experts = cfg.dsf_shared_experts + cfg.dsf_private_experts
        self.task_table = task_table

        self.scene_layer1 = nn.Sequential(
            _init_linear(nn.Linear(cfg.dsf_task_emb + cfg.d_model, self.SCENE_HIDDEN)),
            nn.LayerNorm(self.SCENE_HIDDEN), nn.ReLU())
        self.scene_layer2 = nn.ModuleList(
            _init_linear(nn.Linear(self.SCENE_HIDDEN, self.n_experts)) for _ in range(n_tasks))

        scene_dim = cfg.dsf_task_emb + cfg.d_model
        self.filter = nn.ModuleList(
            nn.Sequential(_init_linear(nn.Linear(in_dim + scene_dim, in_dim)), nn.LayerNorm(in_dim))
            for _ in range(n_tasks))
        dims = [in_dim] + list(cfg.dsf_layer_dims) + [cfg.d_model]
        self.shared = nn.ModuleList(
            ScenarioFactorLayer(dims[i], dims[i + 1], cfg.dsf_shared_experts) for i in range(len(dims) - 1))
        self.n_layers = len(dims) - 1
        self.norms = nn.ModuleList(
            nn.ModuleList(nn.LayerNorm(dims[i + 1]) for i in range(self.n_layers))
            for _ in range(n_tasks)
        )
        self.private_w = nn.ParameterList()
        self.private_b = nn.ParameterList()
        for t in range(n_tasks):
            for i in range(self.n_layers):
                self.private_w.append(nn.Parameter(
                    torch.randn(cfg.dsf_private_experts, dims[i], dims[i + 1]) * (2.0 / math.sqrt(dims[i]))))
                self.private_b.append(nn.Parameter(torch.zeros(cfg.dsf_private_experts, dims[i + 1])))

    def forward(self, x, task_index, profile_vec, mask=None):
        """mask: optional [B, S, 1] float mask (1.0 for live sessions, 0.0 for padded) that zeros
        out padded session slots inside every layer and inside the feature-filtering gate."""
        b = x.size(0)
        task = self.task_table.weight[task_index].unsqueeze(0).expand(b, -1)
        scene_emb = torch.cat([task, profile_vec], dim=-1)
        expert_weight = torch.sigmoid(self.scene_layer2[task_index](self.scene_layer1(scene_emb))) * 2.0

        gate = torch.sigmoid(self.filter[task_index](
            torch.cat([x, scene_emb.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)))
        if mask is not None:
            gate = gate * mask
        x = x * gate
        base = task_index * self.n_layers
        for i, layer in enumerate(self.shared):
            weight, bias = layer.mix(expert_weight, self.private_w[base + i], self.private_b[base + i])
            x = layer.apply_mixed(x, weight, bias, self.norms[task_index][i], mask)
        return x


class IDEmbeddingLayer(nn.Module):

    def __init__(self, table_size, dim):
        super().__init__()
        self.table_size = table_size
        self.id_variable = nn.Parameter(torch.empty(table_size, dim))
        nn.init.normal_(self.id_variable)
        self.register_buffer("const_value", torch.zeros(1, dim))

    @property
    def weight(self):
        return self.id_variable

    def forward(self, idx):
        w = torch.cat([self.id_variable, self.const_value], dim=0)   # [size+1, dim], last row = pad
        i = torch.remainder(idx + self.table_size + 1, self.table_size + 1)  # -1 -> size (zero row)
        return F.embedding(i, w)


class IntHQDemo(nn.Module):
    def __init__(self, cfg, n_ctx, n_sessions):
        super().__init__()
        if cfg.backbone != "inthq":
            raise NotImplementedError("only the 'inthq' backbone is implemented")

        self.cfg = cfg
        self.n_ctx = n_ctx
        self.n_sessions = n_sessions
        self.n_tasks = len(TASKS)

        self.tokens = TokenEmbedding(cfg, n_ctx, n_sessions)
        self.task_table = nn.Embedding(self.n_tasks, cfg.dsf_task_emb)
        nn.init.normal_(self.task_table.weight)

        max_rel = 4 * cfg.max_sessions + N_PROFILE - 1
        self.time_delta = HstuTimeDeltaCross(
            cfg.hstu_time_buckets, cfg.alibi_init_lambda
        )
        self.layers = nn.ModuleList(
            [DualStreamLayer(cfg, max_rel=max_rel, layer_idx=i) for i in range(cfg.n_layers)])
        self.hq = HierarchicalQuery(cfg, cfg.n_layers, cfg.dsf_task_emb)

        head_in = 2 * cfg.d_model
        self.task_head = DSFNet(cfg, head_in, self.n_tasks, self.task_table)

        self.future_table = IDEmbeddingLayer(cfg.n_future_travel, cfg.d_model)

        self.profile_mlp = nn.Sequential(
            _init_linear(nn.Linear(N_PROFILE * cfg.d_model, cfg.d_model)),
            nn.LayerNorm(cfg.d_model), nn.ReLU())

        task_ids = torch.arange(self.n_tasks).repeat(n_sessions)
        self.register_buffer("task_ids", task_ids, persistent=False)
        if cfg.compile_compute and torch.cuda.is_available():
            self.task_head.forward = torch.compile(self.task_head.forward)
            for layer in self.layers:
                layer.forward = torch.compile(layer.forward)
            self.hq.forward = torch.compile(self.hq.forward)

    def split_parameters(self):

        sparse_ids = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding) and module.sparse:
                sparse_ids.add(id(module.weight))
        sparse, dense = [], []
        for p in self.parameters():
            (sparse if id(p) in sparse_ids else dense).append(p)
        return sparse, dense

    @staticmethod
    def _allow(q_pos, k_pos, q_valid, k_valid):

        allowed = (k_pos.unsqueeze(1) <= q_pos.unsqueeze(2)) & k_valid.unsqueeze(1) & q_valid.unsqueeze(2)
        return allowed.float()

    @staticmethod
    def _task_allow(task_pos, task_valid, n_tasks):
        allowed = IntHQDemo._allow(task_pos, task_pos, task_valid, task_valid)
        session = torch.arange(task_pos.size(1), device=task_pos.device) // n_tasks
        same_session = session.unsqueeze(0) == session.unsqueeze(1)
        return allowed * same_session.unsqueeze(0)

    def profile_vector(self, h0):

        emb = h0[:, -N_PROFILE:]
        return self.profile_mlp(emb.reshape(emb.size(0), -1))

    def query_tokens(self, batch_size, device):

        task_ids = self.task_ids.to(device)
        cls_emb = self.tokens.token_type.weight[CLS_TYPE_INDEX]
        per_slot = self.task_table.weight[task_ids] + cls_emb.unsqueeze(0)
        return per_slot.unsqueeze(0).expand(batch_size, -1, -1), task_ids

    def encode(self, batch, h_ctx):
        ctx, task = batch["ctx"], batch["task"]
        h_task, task_ids = self.query_tokens(h_ctx.size(0), h_ctx.device)
        allow_cc = self._allow(ctx["cti"], ctx["cti"], ctx["valid"], ctx["valid"])
        allow_qc = self._allow(task["cti"], ctx["cti"], task["valid"], ctx["valid"])
        allow_qq = self._task_allow(task["cti"], task["valid"], self.n_tasks)

        co = {
            "ctx_gpos": ctx["gpos"], "task_gpos": task["gpos"],
            "ctx_valid": ctx["valid"], "task_valid": task["valid"],
            "allow_cc": allow_cc, "allow_qc": allow_qc, "allow_qq": allow_qq,
            "time_cc": self.time_delta(ctx["time"], ctx["time"], allow_cc),
            "time_qc": self.time_delta(task["time"], ctx["time"], allow_qc),
            "time_qq": self.time_delta(task["time"], task["time"], allow_qq),
        }
        s_ctx, s_task = h_ctx, h_task
        stack = []
        for layer in self.layers:
            s_ctx, s_task, collected = layer(s_ctx, s_task, co)
            stack.append(collected)
        return self.hq(torch.stack(stack, dim=2), task_ids, self.task_table)

    def apply_task_head(self, index, features, profile, mask):
        return self.task_head(features, index, profile, mask)

    def forward(self, batch):
        h0 = self.tokens(batch["ctx"])
        slots = self.encode(batch, h0)
        profile_vec = self.profile_vector(h0)
        task_valid = batch["task"]["valid"]
        out = {}
        for index, name in enumerate(TASKS):
            mask = task_valid[:, index::self.n_tasks].unsqueeze(-1).to(slots.dtype)
            features = slots[:, index::self.n_tasks]
            out[name] = self.apply_task_head(index, features, profile_vec, mask)
        return out

    def candidates(self, entry):
        poi_id = torch.cat([entry["pos_id"].unsqueeze(-1), entry["neg_id"]], dim=-1)
        geo = torch.cat([entry["pos_geo"].unsqueeze(-1), entry["neg_geo"]], dim=-1)
        category = torch.cat([entry["pos_cat"].unsqueeze(-1), entry["neg_cat"]], dim=-1)
        adcode = torch.cat([entry["pos_adcode"].unsqueeze(-1), entry["neg_adcode"]], dim=-1)
        score = torch.cat([entry["pos_score"].unsqueeze(-1), entry["neg_score"]], dim=-1)
        emb = self.tokens.poi(poi_id, geo, category, adcode, score)
        return emb, poi_id, category

    def poi_scores(self, name, out, entry, return_ids=False):
        emb, poi_id, category = self.candidates(entry)

        logits = torch.einsum("bsd,bscd->bsc", out[name], emb)

        neg_inf = torch.finfo(logits.dtype).min / 4
        logits = logits.masked_fill(poi_id < 0, neg_inf)
        if return_ids:
            return logits, category, poi_id
        return logits, category

    def cls_logits(self, name, vec):

        table = self.tokens.travel_mode if name == "how" else self.future_table
        return vec @ table.weight.t()

    @staticmethod
    def _first_valid(valid):
        index = torch.arange(valid.size(1), device=valid.device).expand_as(valid)
        big = torch.full_like(index, index.numel())
        masked = torch.where(valid, index, big)
        first = masked.argmin(dim=1)
        onehot = torch.zeros_like(valid)
        onehot[torch.arange(valid.size(0), device=valid.device), first] = True
        return onehot & valid.any(dim=1, keepdim=True)

    def _holdout(self, batch, valid):
        holdout = batch.get("holdout")
        if holdout is None:
            holdout = self._first_valid(batch["task"]["valid"][:, ::self.n_tasks])
        return holdout.bool() & valid

    def _task_slot_valid(self, batch, name):
        index = TASKS.index(name)
        valid = batch["task"]["valid"][:, index::self.n_tasks].bool()
        source_valid = batch["task"].get("source_valid")
        if source_valid is not None:
            valid = valid & source_valid[:, index::self.n_tasks].bool()
        return valid

    def _poi_base_valid(self, batch, name, entry):
        candidate_valid = (entry["pos_id"] != -1) | (entry["neg_id"] != -1).any(dim=-1)
        return candidate_valid & self._task_slot_valid(batch, name)

    def loss(self, batch):
        out = self.forward(batch)
        labels = batch["labels"]
        losses, metrics = {}, {}

        for name in POI_TASKS:
            entry = labels[name]
            valid = self._poi_base_valid(batch, name, entry)
            train_mask = valid & ~self._holdout(batch, valid)
            if not bool(train_mask.any()):
                continue
            logits, _, _ = self.poi_scores(name, out, entry, return_ids=True)
            logits = logits[train_mask]
            target = torch.zeros_like(logits)
            target[:, 0] = (entry["pos_id"][train_mask] != -1).to(logits.dtype)
            losses[name] = F.cross_entropy(logits, target)
            rank = (logits > logits[:, :1]).sum(-1)
            metrics[name + "/hr@1"] = (rank < 1).float().mean().item()

        for name in ("how", "when"):
            label = labels[name]
            valid = (label != -1) & self._task_slot_valid(batch, name)
            train_mask = valid & ~self._holdout(batch, valid)
            if not bool(train_mask.any()):
                continue
            logits = self.cls_logits(name, out[name])[train_mask]
            target = label[train_mask]
            losses[name] = F.cross_entropy(logits, target)
            metrics[name + "/acc"] = (logits.argmax(-1) == target).float().mean().item()

        total = (torch.stack(list(losses.values())).sum() if losses else
                 torch.zeros((), device=out["where"].device, requires_grad=self.training))
        metrics["loss"] = total.item()
        return total, metrics

    @torch.no_grad()
    def evaluate(self, batch):
        """Paper-table metrics on all base-valid task slots; returns summed counts."""
        out = self.forward(batch)
        labels = batch["labels"]
        stats = {}

        for name in POI_TASKS:
            entry = labels[name]
            eval_mask = self._poi_base_valid(batch, name, entry)
            if not bool(eval_mask.any()):
                continue
            logits, category = self.poi_scores(name, out, entry)
            logits = logits[eval_mask]
            category = category[eval_mask]
            rank = (logits > logits[:, :1]).sum(-1)
            top1 = logits.argmax(-1)
            true_cat = category[:, 0]
            pred_cat = category.gather(1, top1.unsqueeze(-1)).squeeze(-1)
            stats[name] = {
                "n": logits.size(0),
                "hr@1": int((rank < 1).sum()),
                "hr@5": int((rank < 5).sum()),
                "cir": int((pred_cat != true_cat).sum()),
            }

        for name, topk in (("how", 3), ("when", None)):
            label = labels[name]
            eval_mask = (label != -1) & self._task_slot_valid(batch, name)
            if not bool(eval_mask.any()):
                continue
            logits = self.cls_logits(name, out[name])[eval_mask]
            target = label[eval_mask]
            top1 = logits.argmax(-1)
            entry = {"n": logits.size(0), "acc": int((top1 == target).sum())}
            if name == "when":
                entry["abs_err"] = float((top1 - target).abs().sum())
            else:
                topk_hit = (logits.topk(min(topk, logits.size(-1)), dim=-1).indices
                            == target.unsqueeze(-1)).any(-1)
                entry["top3"] = int(topk_hit.sum())
            stats[name] = entry
        return stats

    @torch.no_grad()
    def eval_tensors(self, batch):

        out = self.forward(batch)
        labels = batch["labels"]
        res = {}
        for name in POI_TASKS:
            entry = labels[name]
            valid = self._poi_base_valid(batch, name, entry)
            logits, category = self.poi_scores(name, out, entry)
            res[name] = {
                "prob": logits,
                "mask": valid,
                "poi_label": torch.zeros_like(valid, dtype=torch.long),
                "atag_label": category[..., 0],
                "atag_neg": category[..., 1:],
            }
        for name in ("how", "when"):
            label = labels[name]
            res[name] = {
                "prob": self.cls_logits(name, out[name]),
                "label": label,
                "mask": (label != -1) & self._task_slot_valid(batch, name),
            }
        return res
