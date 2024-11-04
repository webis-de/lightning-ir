"""
Model hub for loading external models.

This module contains mappings and callbacks for external model checkpoints used in the Lightning IR library.

Attributes:
    CHECKPOINT_MAPPING (Dict[str, LightningIRConfig]): Mapping of model checkpoint identifiers to their configurations.
    STATE_DICT_KEY_MAPPING (Dict[str, List[Tuple[str | None, str]]]): Mapping of state dictionary keys for model
        checkpoints.
    POST_LOAD_CALLBACKS (Dict[str, Callable[[LightningIRModel], LightningIRModel]]): Callbacks to be executed after
        loading a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from .config import LightningIRConfig
    from .model import LightningIRModel

CHECKPOINT_MAPPING: Dict[str, LightningIRConfig] = {}
STATE_DICT_KEY_MAPPING: Dict[str, List[Tuple[str | None, str]]] = {}
POST_LOAD_CALLBACKS: Dict[str, Callable[[LightningIRModel], LightningIRModel]] = {}
