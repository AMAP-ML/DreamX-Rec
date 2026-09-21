from transformers import PretrainedConfig
from transformers.models.t5.configuration_t5 import T5Config


class HSTUConfig():
    model_type = "hstu"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        d_model=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        dropout_rate=None,  # 兼容配置文件中的 dropout_rate 参数
        **kwargs,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        # 优先使用 dropout_rate（配置文件常用），否则使用 dropout
        self.dropout = dropout_rate if dropout_rate is not None else dropout
