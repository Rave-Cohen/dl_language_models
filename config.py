import tomllib
from dataclasses import dataclass, fields


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512
    k: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    def merge_from_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    @classmethod
    def from_toml(cls, path: str) -> "GPTConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Use the 'gpt' section if it exists, otherwise assume a flat config structure
        config_dict = data.get("gpt", data)
        if "K" in config_dict and "k" not in config_dict:
            config_dict["k"] = config_dict["K"]

        # Filter the dictionary to only include keys that match dataclass fields
        field_names = {field.name for field in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        return cls(**filtered_dict)
