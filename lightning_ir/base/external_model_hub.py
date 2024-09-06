from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from .config import LightningIRConfig
    from .model import LightningIRModel

CHECKPOINT_MAPPING: Dict[str, LightningIRConfig] = {}
STATE_DICT_KEY_MAPPING: Dict[str, List[Tuple[str | None, str]]] = {}
POST_LOAD_CALLBACKS: Dict[str, Callable[[LightningIRModel], LightningIRModel]] = {}
