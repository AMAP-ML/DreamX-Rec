import warnings
from time import sleep
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers.cache_utils import DynamicCache

from src.utils import pylogger, rich_utils
from functools import partial
from tokenizers.processors import TemplateProcessing
from src.data.loading.components.interfaces import TokenizerConfig

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def print_warnings_for_missing_configs(cfg: DictConfig) -> None:
    _DEFAULT_CONFIGS = [
        "data_loading",
        "model",
        "loss",
        "optim",
        "eval",
    ]
    has_warnings = False
    for config in _DEFAULT_CONFIGS:
        if not cfg.get(config):
            log.warning(
                f"Config {config} was not found in the config tree. Make sure this is expected."
            )
            has_warnings = True
    if has_warnings:
        sleep(3)  # wait for 3 seconds to let the user read the warning


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config_warnings"):
        print_warnings_for_missing_configs(cfg)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        # rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def delete_module(module: torch.nn.Module, module_name: str) -> None:
    """Recursively delete a submodule from a module.

    :param module: the parent module that we want the submodule to be removed from.
    :param module_name: the name of the submodule to be removed.
    :return: None.
    """
    if hasattr(module, module_name):
        delattr(module, module_name)

    for name, submodule in module.named_children():
        delete_module(submodule, module_name)


def find_module_shape(
    module: torch.nn.Module, module_name: str
) -> Optional[torch.Size]:
    """Recursively find a submodule in a module and return its shape.

    :param module: the parent module that we want the submodule to be removed from.
    :param module_name: the name of the submodule to be removed.
    :return: the shape of the module if it exists.
    """
    if hasattr(module, module_name):
        return getattr(module, module_name).weight.shape

    for name, submodule in module.named_children():
        shape = find_module_shape(submodule, module_name)
        if shape:
            return shape
    return None


def reset_parameters(module: torch.nn.Module) -> None:
    """Reset the parameters of a given module.

    :param module: the module whose parameters will be reset.
    :return: None.
    """

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for layer in module.children():
            reset_parameters(layer)


def get_var_if_not_none(value: Optional[Any], default_value: Any) -> Any:
    """
    :return: value if value is not None, else default_value
    Note that when value is:
        Boolean: False is not None
        Int: 0 is not None
        List: An empty list is not None
        Tensor: A tensor with all zeros is not None
    """
    return value if value is not None else default_value


def get_class_name_str(class_definition: Any) -> str:
    """
    Args:
        class_definition: The class definition.

    Returns:
        The fully qualified name of the given class.
    """
    return ".".join([class_definition.__module__, class_definition.__name__])


def has_class_object_inside_list(obj_list: list, class_type: Any) -> bool:
    """
    Args:
        obj_list: List of objects.
        class_type: The class type to check.

    Returns:
        True if the list contains an object of the given class type.
    """
    return any(isinstance(obj, class_type) for obj in obj_list)


def convert_legacy_kv_cache_to_dynamic(
    kv_cache: Union[DynamicCache, Tuple[torch.Tensor]]
) -> DynamicCache:
    """
    Converts a legacy key-value cache (Tuple of tensors) to a dynamic cache.

    Args:
        kv_cache: The key-value cache.

    Returns:
        The dynamic cache.
    """
    if isinstance(kv_cache, DynamicCache):
        return kv_cache

    return DynamicCache.from_legacy_cache(kv_cache)


def get_parent_module_and_attr(
    model: torch.nn.Module, module_name: str
) -> Tuple[torch.nn.Module, str]:
    """
    Get the parent module and attribute name for a given module name.

    Args:
        model (torch.nn.Module): The model containing the module.
        module_name (str): The full name of the module.

    Returns:
        Tuple[torch.nn.Module, str]: The parent module and the attribute name.
    """
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def lightning_precision_to_dtype(precision: str) -> torch.dtype:
    """
    Convert Lightning precision identifier to PyTorch dtype.

    Args:
        precision (str): The precision identifier used in Lightning.
                         Expected values include '32', '32-true', '16', '16-mixed', 'bf16', '64', 'half'.

    Returns:
        torch.dtype: The corresponding PyTorch dtype.

    Raises:
        ValueError: If an unsupported precision type is provided.
    """
    # Mapping from Lightning precision identifiers to PyTorch dtypes
    precision_map = {
        "32": torch.float32,  # Single precision (float32)
        "32-true": torch.float32,  # Also maps to float32, useful for clarity when specifying defaults
        "64": torch.float64,  # Double precision
        "16": torch.float16,  # Half precision
        "16-mixed": torch.float16,  # Mixed precision typically uses torch.float16
        "bf16": torch.bfloat16,  # BFloat16 precision
        "half": torch.float16,  # Alias for half precision
    }

    if precision in precision_map:
        return precision_map[precision]
    else:
        raise ValueError(
            f"Unsupported precision type: '{precision}'. "
            "Supported precision types are: '32', '32-true', '64', '16', '16-mixed', 'bf16', 'half'."
        )

def load_tokenize(config: TokenizerConfig) -> Any:
    """Load tokenizer and return a partial function for tokenization."""
    tokenizer = config.tokenizer
    if hasattr(config, "special_tokens"):
        tokenizer.add_special_tokens(config.special_tokens)
    if config.postprocess_eos_token:
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A " + tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)],
        )
    tokenize = partial(
        tokenizer.encode_plus,
        max_length=config.max_length,
        padding=config.padding,
        truncation=config.truncation,
        add_special_tokens=config.add_special_tokens,
        return_tensors="pt",
    )
    return tokenize

def sample_gumbel(shape: Tuple, device: torch.device, eps=1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, logits.device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample


def unique_ids_and_sids(input_ids, attention_mask, input_sids=None):
    """
    Returns:
      unique_input_ids: [U, 1]
      inverse_indices: [N_valid]  每个 valid 位置对应 unique 的索引
      unique_input_sids: [U, 4] or None
    """
    if attention_mask.dim() == 3:
        attention_mask = attention_mask.squeeze(-1)  # [B,S]

    # 展平有效位置
    valid_mask = attention_mask.bool()               # [B,S]
    valid_ids = input_ids[valid_mask]                # [N_valid, 1]

    # 对 [N_valid,1] 先 squeeze 成 [N_valid] 做 unique
    valid_ids_1d = valid_ids.squeeze(-1)             # [N_valid]

    # unique（保留首次出现顺序）
    unique_1d, inverse_indices = torch.unique(
        valid_ids_1d, return_inverse=True, sorted=False
    )                                                # unique_1d: [U], inverse_indices: [N_valid]

    # 计算每个 unique 的“首次出现位置” first_pos: [U]
    U = unique_1d.numel()
    first_pos = torch.full((U,), valid_ids_1d.numel(),
                           device=valid_ids_1d.device, dtype=torch.long)
    first_pos.scatter_reduce_(
        0,
        inverse_indices,
        torch.arange(valid_ids_1d.numel(), device=valid_ids_1d.device),
        reduce="amin",
        include_self=True
    )

    unique_input_ids = unique_1d.unsqueeze(-1)       # [U,1]

    if input_sids is not None:
        valid_sids = input_sids[valid_mask]          # [N_valid, 4]
        unique_input_sids = valid_sids[first_pos]    # [U,4]
    else:
        unique_input_sids = None

    return unique_input_ids, inverse_indices, unique_input_sids