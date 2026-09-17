from dataclasses import dataclass, replace

TASKS = ("where", "how", "when", "via")
TASK_TOKEN_OFFSET = {"where": 0, "how": 0, "when": 0, "via": 1}
POI_TASKS = ("where", "via")

N_PROFILE = 6
TOK_PER_SESSION = 3

HASH_PRIMES = (5008057, 5008121, 5008259, 5008433)


@dataclass
class Config:
    max_actions: int = None
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

    # ---- encoder ----
    d_model: int = 96
    n_layers: int = 16
    hstu_time_buckets: int = 256
    alibi_init_lambda: float = 1e-2
    emb_init_std: float = 1e-3

    # ---- embedding backend ----
    # "auto"  -> use RecIS if importable, otherwise pure PyTorch
    # "torch" -> pure PyTorch implementation
    # "recis" -> RecIS implementation
    embedding_backend: str = "recis"

    # ---- optimization ----
    epochs: int = 1
    batch_size: int = 64
    lr: float = 8e-4
    weight_decay: float = 1e-6
    seed: int = 42

    backbone: str = "inthq"

    # ---- cross-layer task gate ----
    hq_attn_dim: int = 64
    hq_num_heads: int = 1

    # ---- task embedding / DSFNet ----
    dsf_shared_experts: int = 20
    dsf_private_experts: int = 180
    dsf_task_emb: int = 96
    dsf_layer_dims: tuple = (256, 128)

    # ---- prediction tables ----
    n_travel_mode: int = 6
    n_future_travel: int = 49

    # torch.compile
    compile_compute: bool = True

    # ---- sparse embedding update (pure-torch equivalent of the RecIS sparse optimizer) ----
    sparse_embedding: bool = True

    def local_example(self, embedding_backend="torch"):
        """Shrink the id spaces so the bundled 100-case sample runs quickly.

        embedding_backend:
            "torch" -> pure-PyTorch multi-hash nn.Embedding (runs on CPU or a single GPU).
            "recis" -> RecIS dynamic hash tables (needs a CUDA device + torch.distributed).
        """
        return replace(
            self,
            hash_primes=(20011, 20021, 20023, 20029),
            dsf_shared_experts=2,
            dsf_private_experts=2,
            embedding_backend=embedding_backend,
            sparse_embedding=False,
            compile_compute=False,
        )
