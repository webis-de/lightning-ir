from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Set, Type

if TYPE_CHECKING:
    from . import LightningIRTokenizer


class LightningIRConfig(ABC):
    """The configuration class to instantiate a LightningIR model. Acts as a mixin for the
    transformers.PretrainedConfig_ class.

    .. _transformers.PretrainedConfig: https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig
    """

    model_type = "lightning-ir"
    """Model type for the configuration."""
    backbone_model_type: str | None = None
    """Backbone model type for the configuration. Set by :func:`LightningIRModelClassFactory`."""

    @property
    @abstractmethod
    def tokenizer_class(self) -> Type[LightningIRTokenizer] | None:
        """Tokenizer class for the configuration. Needs to be set in derived config."""
        ...

    TOKENIZER_ARGS: Set[str] = {"query_length", "doc_length"}
    """Arguments for the tokenizer."""
    ADDED_ARGS: Set[str] = TOKENIZER_ARGS
    """Arguments added to the configuration."""

    def __init__(self, *args, query_length: int = 32, doc_length: int = 512, **kwargs):
        """Initializes the configuration.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        """
        super().__init__(*args, **kwargs)
        self.query_length = query_length
        self.doc_length = doc_length

    def to_added_args_dict(self) -> Dict[str, Any]:
        """Outputs a dictionary of the added arguments.

        :return: Added arguments
        :rtype: Dict[str, Any]
        """
        return {arg: getattr(self, arg) for arg in self.ADDED_ARGS if hasattr(self, arg)}

    def to_tokenizer_dict(self) -> Dict[str, Any]:
        """Outputs a dictionary of the tokenizer arguments.

        :return: Tokenizer arguments
        :rtype: Dict[str, Any]
        """
        return {arg: getattr(self, arg) for arg in self.TOKENIZER_ARGS}

    def to_dict(self) -> Dict[str, Any]:
        """Overrides the `to_dict` method to include the added arguments and the backbone model type.

        :return: Configuration dictionary
        :rtype: Dict[str, Any]
        """
        if hasattr(super(), "to_dict"):
            output = getattr(super(), "to_dict")()
        else:
            output = self.to_added_args_dict()
        if self.__class__.model_type is not None:
            output["backbone_model_type"] = self.__class__.backbone_model_type
        return output
