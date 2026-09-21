from dataclasses import dataclass, replace

TASKS = ("where", "how", "when", "via")
TASK_TOKEN_OFFSET = {"where": 0, "how": 0, "when": 0, "via": 1}
POI_TASKS = ("where", "via")

N_PROFILE = 6
TOK_PER_SESSION = 3
HASH_PRIMES = (5008057, 5008121, 5008259, 5008433)


@dataclass
class Config:
    max_sessions: int = 40

    hash_primes: tuple = HASH_PRIMES
    sub_dim: int = 24
    seq_type_vocab: int = 5
    action_type_vocab: int = 8
    adcode_vocab: int = 3274
    category_vocab: int = 21
    weather_vocab: int = 21
    weather_dim: int = 48
    profile_vocab: int = 47

    d_model: int = 96
    n_layers: int = 3
    hstu_time_buckets: int = 256
    alibi_init_lambda: float = 1e-2
    emb_init_std: float = 1e-3

    embedding_backend: str = "recis"
    sparse_embedding: bool = True
    compile_compute: bool = True

    epochs: int = 1
    batch_size: int = 64
    lr: float = 8e-4
    weight_decay: float = 1e-6
    seed: int = 42

    backbone: str = "inttravel"

    hc_expansion_rate: int = 2
    hc_use_dynamic: bool = True
    hc_use_tanh: bool = True
    hc_use_scaling_factor: bool = True
    hc_use_prenorm_init: bool = True
    hc_task_gate_am_ar: bool = True
    hc_task_gate_b: bool = False
    hc_task_gating_mode: str = "scaling"

    tsg_gate_type: str = "scalar"
    tsg_shared_across_layers: bool = False

    dsf_shared_experts: int = 20
    dsf_private_experts: int = 180
    dsf_task_emb: int = 32
    dsf_layer_dims: tuple = (256, 128)

    n_travel_mode: int = 6
    n_future_travel: int = 49

    def local_example(self, embedding_backend="torch"):
        return replace(
            self,
            hash_primes=(20011, 20021, 20023, 20029),
            dsf_shared_experts=2,
            dsf_private_experts=2,
            embedding_backend=embedding_backend,
            sparse_embedding=False,
            compile_compute=False,
        )
