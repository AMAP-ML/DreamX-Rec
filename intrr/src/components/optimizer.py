from torch.optim import Optimizer


class PassThroughOptimizer(Optimizer):
    """
    A dummy PyTorch optimizer that does nothing during the step() function.

    This can be used for testing purposes or for training Lightning modules with manual
    parameter updates. In Lightning, we need to call `opt.step()` so that
    trainer.global_step is incremented, even if we are doing manual optimization.
    """

    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(PassThroughOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        return None

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def state_dict(self):
        # needed for lightning checkpointing
        return {}

    def load_state_dict(self, state_dict):
        # needed for lightning checkpointing
        pass
