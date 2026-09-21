import torch
from pytorch_lightning import LightningModule


def scale_loss_by_world_size_for_initialization_training_loop(
    model: LightningModule,
    loss: torch.Tensor,
    world_size: int,
    is_initialized: bool = True,
    initalization_optimizer_lr: float = 0.5,
    initalization_optimizer: torch.optim.Optimizer = torch.optim.SGD,
):
    """
    Training loop that scales the loss by the number of GPUs used for training during
    model initialization.

    This is used for training models with DDP that require special initialization
    computed based on the data, such as KMeans. To use this function to initialize a
    model:

    1. Ensure that the model parameters are set to zero at the very start of training.

    2. On one device, compute the desired initial parameters based on the data, and
    compute the loss as the squared distance between the model's parameters and the
    desired initial parameters. We use only one device to compute the initialization
    because different devices may have different data, and the average initialization
    across devices may not be a reasonable initialization.

    3. Set the loss on other devices to zero.

    4. Feed the loss to this training loop with the `is_initialized=False`. This will
    tell this function to scale the loss by the world size, which is necessary for
    the following reason:

    Without loss rescaling, the losses are:
    - Device 1: || model.parameters - desired_initial_parameters||^2
    - Device 2-world_size: 0
    The gradients are (recalling that model.parameters are all zero):
    - Device 1: -2 * desired_initial_parameters
    - Device 2-world_size: 0
    Recall that DDP will average the gradients across all devices. At this point, the
    average gradient is:
      -2 * desired_initial_parameters / world_size
    which means that with SGD optimizer with learning rate 0.5, the model's parameters
    will be `desired_initial_parameters / world_size` after the update. To avoid this,
    we scale the loss by the world size so that the average gradient across all devices
    is -2 * desired_initial_parameters, . This means that the model's parameters will be
    updated to the desired initial parameters in one step on all devices via DDP.
    Note that this is only used for the initialization step. After the model is
    initialized, we can use the default training loop without loss rescaling because the
    gradients on all devices are non-trivial.

    5. Ensure that the optimizer is SGD with learning rate 0.5 so that the model's
    parameters are updated to the desired initial parameters in one step on all devices
    via DDP.

    6. After the model is initialized in this step, set the `is_initialized` flag to
    `True` to use the default training loop, as now we can compute losses on all devices
    and no longer need to scale the loss by the world size.

    Args:
        trainer: The trainer object.
        loss: The loss value. If the model is not yet initialized, this loss
            should be the squared distance between the model's parameters, which should
            have value zero at this point, and the desired initial parameters.
        world_size: The number of GPUs used for training.
        is_initialized: A boolean indicating whether the model is already initialized.
    """
    if not is_initialized:
        # Perform special initialization
        # We scale the loss by the world size for proper initialization
        opt = initalization_optimizer(model.parameters(), lr=initalization_optimizer_lr)
        loss = loss * world_size
    else:
        # Use the default training loop
        opt = model.optimizers()
    opt.zero_grad()
    model.manual_backward(loss)
    opt.step()
