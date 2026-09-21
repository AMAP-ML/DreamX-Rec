import ast
import operator as op
from typing import Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

""""
Hydra allows for custom resolvers, which are functions that can be used to resolve values in the config.
For example, one can manipulate strings or apply simple python functions to the config values.

"""


def remove_chars_from_string(s: str, chars: str) -> str:
    """Removes all occurrences of `chars` from `s`.

    :param s: The input string.
    :param chars: The characters to remove from `s`.

    :return: The string `s` with all occurrences of `chars` removed.
    """
    return s.translate(str.maketrans("", "", chars))


def conditional_expression(
    condition_expression, value_if_true, value_if_false, **kwargs
):
    """
    A generic resolver that evaluates a condition expression based on config values.
    """
    try:
        # Evaluate the condition expression with the config context
        result = eval(condition_expression, {}, kwargs)
        return value_if_true if result else value_if_false
    except Exception as e:
        raise ValueError(
            f"Error evaluating condition: {condition_expression}. Error: {e}"
        )


def extract_fields_from_list_of_dicts(
    list_of_dicts: ListConfig,
    key: str,
    default: str = None,
    filter_key: str = None,
    filter_value: str = None,
) -> ListConfig:
    """
    Extracts a list of values from a list of dictionaries based on a key, with an optional filter condition.

    :param list_of_dicts: The list of dictionaries to extract values from.
    :param key: The key to extract values for.
    :param default: The default value to use if the key is not found in a dictionary.
    :param filter_key: The key to filter dictionaries by.
    :param filter_value: The value that the filter_key should have for a dictionary to be included.

    Example:
    Given a list of dictionaries:
    [
        {"name": "feature1", "is_sparse": True},
        {"name": "feature2", "is_sparse": False},
        {"name": "feature3"},
    ]

    extract_fields_from_list_of_dicts(features, "name")
    will return ["feature1", "feature2", "feature3"]

    extract_fields_from_list_of_dicts(features, "is_sparse", default=False)
    will return [True, False, False]

    extract_fields_from_list_of_dicts(features, "name", default=False, filter_key="is_sparse", filter_value="True")
    will return ["feature1"]


    :return: A ListConfig of extracted values.
    """
    if filter_key and filter_value:
        filtered_dicts = [
            d for d in list_of_dicts if d.get(filter_key) == eval(filter_value)
        ]
    else:
        filtered_dicts = list_of_dicts

    return ListConfig([d.get(key, default) for d in filtered_dicts])


def create_map_from_list_of_dicts(
    list_of_dicts: ListConfig, key: str, value: Optional[str] = None
) -> DictConfig:
    """
    Creates a dictionary from a list of dictionaries based on the key and value.
    For example, if a feature has a name and an attribute name dim, this function can be used to create a mapping
    from the feature name to the attribute:
    create_map_from_list_of_dicts(features, "name", "dim")
    If value is not provided, the function will return a dictionary with the key as the key and the value as the dictionary.
    create_map_from_list_of_dicts(features, "name")
    will return {"feature1": {"name": "feature1", "dim": 10}, "feature2": {"name": "feature2", "dim": 20}, "feature3": {"name": "feature3"}}
    """
    if value is None:
        return DictConfig({d[key]: d for d in list_of_dicts if key in d})

    return DictConfig(
        {d[key]: d[value] for d in list_of_dicts if key in d and value in d}
    )


def math_eval(expression: str) -> float:
    """
    Evaluate a mathematical expression given as a string.

    Examples:

    ${math_eval:"2^6"} returns 64

    ${math_eval:"1 + 2*3**4 / (5 + -6)"} returns -161.0

    dim_1: 32
    dim_2: 96
    ${math_eval:${dim_1}+${dim_2}} returns 128
    """
    # Supported operators
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.BitXor: op.xor,
        ast.USub: op.neg,
    }

    def eval_(node):
        # Recursively evaluate the AST nodes
        match node:
            case ast.Constant(value) if isinstance(value, int):
                return value  # integer
            case ast.BinOp(left, op, right):
                return operators[type(op)](eval_(left), eval_(right))
            case ast.UnaryOp(op, operand):  # e.g., -1
                return operators[type(op)](eval_(operand))
            case _:
                raise TypeError(node)

    return eval_(ast.parse(expression, mode="eval").body)


def remove_item_from_list(input_list: ListConfig, item_to_remove: str) -> ListConfig:
    """
    Removes all occurrences of a specific item from a list.
    :param input_list: The input list to remove items from.
    :param item_to_remove: The item to remove from the list.
    :return: A ListConfig with the specified item removed.
    """
    return ListConfig([item for item in input_list if item != item_to_remove])


# resolvers need to be registered to be accessible during config composition.
# The resolver name is the function name without the type annotations.
OmegaConf.register_new_resolver("remove_chars_from_string", remove_chars_from_string)
OmegaConf.register_new_resolver("conditional_expression", conditional_expression)
OmegaConf.register_new_resolver(
    "extract_fields_from_list_of_dicts", extract_fields_from_list_of_dicts
)
OmegaConf.register_new_resolver(
    "create_map_from_list_of_dicts", create_map_from_list_of_dicts
)
OmegaConf.register_new_resolver("math_eval", math_eval)
OmegaConf.register_new_resolver("remove_item_from_list", remove_item_from_list)
