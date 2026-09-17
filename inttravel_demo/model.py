import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from inthq_demo.model import (
    DSFNet,
    IDEmbeddingLayer,
    IntHQDemo,
    TokenEmbedding,
    _init_linear,
)

from .config import N_PROFILE, TASKS, TASK_TOKEN_OFFSET


class HstuCore(nn.Module):
    def __init__(self, cfg, max_rel):
        super().__init__()
        self.uvqk = _init_linear(nn.Linear(cfg.d_model, 4 * cfg.d_model))
        self.out = _init_linear(nn.Linear(cfg.d_model, cfg.d_model))
        self.av_norm = nn.LayerNorm(cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.rel = nn.Parameter(torch.randn(2 * max_rel + 1) * 0.02)
        self.alibi = nn.Parameter(torch.tensor(float(cfg.alibi_init_lambda)))
        self.max_rel = max_rel
        self.time_buckets = cfg.hstu_time_buckets

    def forward(self, inputs, cti, gpos, time, valid):
        uvqk = F.silu(self.uvqk(inputs))
        u, v, q, k = uvqk.chunk(4, dim=-1)
        similarity = q @ k.transpose(-1, -2)
        rel_index = (gpos.unsqueeze(1) - gpos.unsqueeze(2) + self.max_rel).clamp(
            0, 2 * self.max_rel
        )
        similarity = similarity + self.rel[rel_index]
        delta = (time.unsqueeze(2) - time.unsqueeze(1)).abs().clamp(min=1).float()
        bucket = (torch.log(delta) / 0.301).clamp(0, self.time_buckets).detach()
        allow = (
            (cti.unsqueeze(1) <= cti.unsqueeze(2))
            & valid.unsqueeze(1)
            & valid.unsqueeze(2)
        ).to(similarity.dtype)
        timestamp_valid = valid & time.ge(0)
        timestamp_allow = (
            timestamp_valid.unsqueeze(1) & timestamp_valid.unsqueeze(2)
        ).to(similarity.dtype)
        similarity = F.silu(similarity - self.alibi * bucket * timestamp_allow) * allow
        count = similarity.ne(0).sum(dim=-1, keepdim=True).clamp(min=1).float()
        attended = (similarity / count) @ v
        output = self.out_norm(self.out(self.av_norm(attended) * u))
        return output * valid.unsqueeze(-1).to(output.dtype)


class HyperConnectionWeights(nn.Module):
    def __init__(self, cfg, layer_index):
        super().__init__()
        self.n = cfg.hc_expansion_rate
        self.d = cfg.d_model
        self.use_dynamic = cfg.hc_use_dynamic
        self.use_tanh = cfg.hc_use_tanh
        self.use_scaling = cfg.hc_use_scaling_factor

        am = torch.zeros(self.n, 1)
        am[layer_index % self.n, 0] = 1.0
        self.static_am = nn.Parameter(am)
        self.static_ar = nn.Parameter(torch.eye(self.n))
        self.static_b = nn.Parameter(torch.ones(1, self.n))

        if self.use_dynamic:
            self.norm = nn.LayerNorm(self.d)
            self.w_am = nn.Parameter(torch.zeros(self.n * self.d, self.n))
            self.w_ar = nn.Parameter(torch.zeros(self.n * self.d, self.n * self.n))
            self.w_b = nn.Parameter(torch.zeros(self.n * self.d, self.n))
            if self.use_scaling:
                self.alpha = nn.Parameter(torch.tensor(0.01))
                self.beta = nn.Parameter(torch.tensor(0.01))

    def forward(self, inputs):
        batch, length = inputs.shape[:2]
        if not self.use_dynamic:
            am = self.static_am.view(1, 1, self.n, 1).expand(batch, length, -1, -1)
            ar = self.static_ar.view(1, 1, self.n, self.n).expand(batch, length, -1, -1)
            b = self.static_b.view(1, 1, 1, self.n).expand(batch, length, -1, -1)
            return am, ar, b

        flat = self.norm(inputs).reshape(batch * length, self.n * self.d)
        am = (flat @ self.w_am).view(batch, length, self.n, 1)
        ar = (flat @ self.w_ar).view(batch, length, self.n, self.n)
        b = (flat @ self.w_b).view(batch, length, 1, self.n)
        if self.use_tanh:
            am, ar, b = torch.tanh(am), torch.tanh(ar), torch.tanh(b)
        if self.use_scaling:
            am, ar, b = am * self.alpha, ar * self.alpha, b * self.beta
        return (
            am + self.static_am.view(1, 1, self.n, 1),
            ar + self.static_ar.view(1, 1, self.n, self.n),
            b + self.static_b.view(1, 1, 1, self.n),
        )


class TaskScaling(nn.Module):
    def __init__(self, task_dim, width):
        super().__init__()
        self.projection = nn.Linear(task_dim, width)
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 3.0)

    def forward(self, task_table):
        return torch.sigmoid(self.projection(task_table))


class MultiTaskHyperConnectionBlock(nn.Module):
    def __init__(self, cfg, layer_index, max_rel):
        super().__init__()
        self.n = cfg.hc_expansion_rate
        self.weights = HyperConnectionWeights(cfg, layer_index)
        self.task_scaling = TaskScaling(cfg.dsf_task_emb, self.n)
        self.core = HstuCore(cfg, max_rel)

    def forward(self, inputs, task_table, cti, gpos, time, valid):
        am, ar, b = self.weights(inputs)
        task_weight = self.task_scaling(task_table).mean(dim=0).view(1, 1, self.n, 1)
        h0 = torch.matmul((am * task_weight).transpose(-1, -2), inputs).squeeze(-2)
        residual = torch.matmul((ar * task_weight).transpose(-1, -2), inputs)

        message = self.core(h0, cti, gpos, time, valid)
        broadcast = torch.matmul(b.transpose(-1, -2), message.unsqueeze(2))
        return (residual + broadcast) * valid.unsqueeze(-1).unsqueeze(-1).to(inputs.dtype)


class IntTravelEncoder(nn.Module):
    def __init__(self, cfg, max_rel):
        super().__init__()
        self.width = cfg.hc_expansion_rate
        self.blocks = nn.ModuleList(
            MultiTaskHyperConnectionBlock(cfg, index, max_rel)
            for index in range(cfg.n_layers)
        )
        self.output_norms = nn.ModuleList(
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        )

    def forward(self, inputs, task_table, cti, gpos, time, valid):
        state = inputs.unsqueeze(2).expand(-1, -1, self.width, -1).contiguous()
        outputs = []
        for block, norm in zip(self.blocks, self.output_norms):
            state = block(state, task_table, cti, gpos, time, valid)
            outputs.append(norm(state.sum(dim=2)))
        return torch.stack(outputs, dim=2)


class TaskSpecificGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.tsg_gate_type != "scalar" or cfg.tsg_shared_across_layers:
            raise NotImplementedError("the paper configuration uses private scalar layer gates")
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(cfg.dsf_task_emb, 1), nn.Sigmoid())
            for _ in range(cfg.n_layers)
        )

    def forward(self, layer_outputs, task_embedding):
        gated = [
            layer_outputs[:, :, index] * gate(task_embedding).view(1, 1, 1)
            for index, gate in enumerate(self.gates)
        ]
        if len(gated) == 1:
            return gated[0]
        return torch.cat([gated[-1], torch.stack(gated[:-1], dim=0).mean(dim=0)], dim=-1)


class IntTravelDemo(IntHQDemo):
    def __init__(self, cfg, n_ctx, n_sessions):
        nn.Module.__init__(self)
        if cfg.backbone != "inttravel":
            raise NotImplementedError("only the 'inttravel' backbone is implemented")
        self.cfg = cfg
        self.n_ctx = n_ctx
        self.n_sessions = n_sessions
        self.n_tasks = len(TASKS)

        self.tokens = TokenEmbedding(cfg, n_ctx, n_sessions)
        self.task_table = nn.Embedding(self.n_tasks, cfg.dsf_task_emb)
        nn.init.normal_(self.task_table.weight)
        max_rel = 4 * cfg.max_sessions + N_PROFILE - 1
        self.encoder = IntTravelEncoder(cfg, max_rel=max_rel)
        self.task_gates = nn.ModuleDict(
            {name: TaskSpecificGate(cfg) for name in TASKS}
        )
        task_head_dim = cfg.d_model if cfg.n_layers == 1 else 2 * cfg.d_model
        self.task_head = DSFNet(
            cfg, task_head_dim, self.n_tasks, self.task_table
        )
        self.future_table = IDEmbeddingLayer(cfg.n_future_travel, cfg.d_model)
        self.profile_mlp = nn.Sequential(
            _init_linear(nn.Linear(N_PROFILE * cfg.d_model, cfg.d_model)),
            nn.LayerNorm(cfg.d_model),
            nn.ReLU(),
        )

        session = torch.arange(n_sessions)
        host = {
            name: session * 3 + TASK_TOKEN_OFFSET[name]
            for name in TASKS
        }
        for name, index in host.items():
            self.register_buffer("host_" + name, index, persistent=False)

    @property
    def layers(self):
        return self.encoder.blocks

    def forward(self, batch):
        ctx = batch["ctx"]
        embeddings = self.tokens(ctx)
        encoded = self.encoder(
            embeddings,
            self.task_table.weight,
            ctx["cti"],
            ctx["gpos"],
            ctx["time"],
            ctx["valid"],
        )
        profile = self.profile_vector(embeddings)
        task_valid = batch["task"]["valid"]
        outputs = {}
        for task_index, name in enumerate(TASKS):
            index = getattr(self, "host_" + name)
            selected = encoded.index_select(1, index)
            features = self.task_gates[name](
                selected, self.task_table.weight[task_index]
            )
            mask = task_valid[:, task_index::self.n_tasks].unsqueeze(-1).to(
                features.dtype
            )
            outputs[name] = self.apply_task_head(
                task_index, features, profile, mask
            )
        return outputs
