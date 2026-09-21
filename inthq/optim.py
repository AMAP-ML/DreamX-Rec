import torch


class SparseAdamW(torch.optim.SparseAdam):
    def __init__(self, params, *, lr, weight_decay):
        super().__init__(params, lr=lr)
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        if self.weight_decay:
            for group in self.param_groups:
                decay = 1.0 - group["lr"] * self.weight_decay
                for parameter in group["params"]:
                    if parameter.grad is None or not parameter.grad.is_sparse:
                        continue
                    rows = parameter.grad.coalesce().indices()[0].unique()
                    parameter.data.index_copy_(
                        0,
                        rows,
                        parameter.data.index_select(0, rows) * decay,
                    )
        return super().step(closure)


def build_optimizers(model, cfg):
    if model.tokens.id_backend == "recis":
        from recis.nn.modules.hashtable import filter_out_sparse_param
        from recis.optim import NamedAdamWTF, SparseAdamWTF

        sparse_parameters = filter_out_sparse_param(model)
        dense_parameters = list(model.parameters())
        dense_optimizer = NamedAdamWTF(
            model.named_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        sparse_optimizer = SparseAdamWTF(
            sparse_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        return dense_optimizer, sparse_optimizer, dense_parameters

    sparse_parameters, dense_parameters = model.split_parameters()
    dense_optimizer = torch.optim.AdamW(
        dense_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sparse_optimizer = None
    if sparse_parameters:
        sparse_optimizer = SparseAdamW(
            sparse_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    return dense_optimizer, sparse_optimizer, dense_parameters
