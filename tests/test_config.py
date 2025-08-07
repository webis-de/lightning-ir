from pathlib import Path

from transformers import AutoConfig

from lightning_ir.base.config import LightningIRConfig
from lightning_ir.base.module import LightningIRModule


def test_serialize_deserialize(module: LightningIRModule, tmp_path: Path):
    config = module.model.config
    config_class = module.model.config_class
    save_dir = str(tmp_path / config_class.model_type)
    config.save_pretrained(save_dir)
    new_configs = [
        config.__class__.from_pretrained(save_dir),
        config.__class__.__bases__[0].from_pretrained(save_dir),
        LightningIRConfig.from_pretrained(save_dir),
        AutoConfig.from_pretrained(save_dir),
    ]
    for new_config in new_configs:
        for key, value in config.__dict__.items():
            if key in (
                "torch_dtype",
                "_name_or_path",
                "_commit_hash",
                "transformers_version",
                "_attn_implementation_autoset",
                "_attn_implementation_internal",
            ):
                continue
            assert getattr(new_config, key) == value
