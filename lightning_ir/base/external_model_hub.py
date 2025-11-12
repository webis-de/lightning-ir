"""
Model hub for loading external models.

This module contains mappings and callbacks for external model checkpoints used in the Lightning IR library.

Attributes:
    CHECKPOINT_MAPPING (dict[str, LightningIRConfig]): Mapping of model checkpoint identifiers to their configurations.
    BACKBONE_MAPPING (dict[str, type]): Mapping of model checkpoint identifiers to their backbone model classes.
    STATE_DICT_KEY_MAPPING (dict[str, list[tuple[str | None, str]]]): Mapping of state dictionary keys for model
        checkpoints.
    POST_LOAD_CALLBACKS (dict[str, Callable[[LightningIRModel], LightningIRModel]]): Callbacks to be executed after
        loading a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .config import LightningIRConfig
    from .model import LightningIRModel

CHECKPOINT_MAPPING: dict[str, LightningIRConfig] = {}
BACKBONE_MAPPING: dict[str, type] = {}
STATE_DICT_KEY_MAPPING: dict[str, list[tuple[str | None, str]]] = {}
POST_LOAD_CALLBACKS: dict[str, Callable[[LightningIRModel], LightningIRModel]] = {}
