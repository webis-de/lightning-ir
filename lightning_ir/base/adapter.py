"""
Adapter module for Lightning IR models.

This module provides LoRA adapter support for Lightning IR models using the PEFT library.
The adapter functionality is optional and only enabled when explicitly configured.
"""
from __future__ import annotations

try:
    from peft import LoraConfig, PeftModel, get_peft_model, PeftConfig, TaskType
    
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False



class LightningIRAdapterMixin:
    """Mixin class that adds LoRA adapter functionality to Lightning IR models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter_initialized = False
        self._adapter_enabled = False
        self._adapter_config = None
        self._is_peft_model = False

    def init_adapters(self, adapter_config: LoraConfig) -> None:
        """Enable LoRA adapters on the model.
        
        Args:
            adapter_config: Configuration for the LoRA adapter.
        
        Raises:
            ImportError: If PEFT is not available.
            ValueError: If adapters are already enabled.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT is required for adapter functionality. "
                "Install it with: pip install lightning-ir[adapters]"
            )

        if self._adapter_enabled:
            raise ValueError("Adapters are already enabled on this model")

        self._adapter_config = adapter_config
        peft_model = get_peft_model(self, adapter_config)

        for name, module in peft_model.named_children():
            if hasattr(self, name) and name != 'base_model':
                original_module = getattr(self, name)
                if original_module is not module:  # Only set if it's actually different
                    setattr(self, name, module)

        self._adapter_initialized = True
        self._adapter_enabled = True
        self._is_peft_model = True


    def disable_adapters(self) -> None:
        """Disable LoRA adapters."""
        if not self._adapter_enabled:
            return
        if hasattr(self, 'disable_adapter_layers'):
            self.disable_adapter_layers()
        elif hasattr(self, 'disable_adapter'):
            self.disable_adapter()

    def enable_adapters(self) -> None:
        """(Re-)Enable LoRA adapters."""
        if self._adapter_enabled:
            return
        if hasattr(self, 'enable_adapter_layers'):
            self.enable_adapter_layers()
        elif hasattr(self, 'enable_adapter'):
            self.enable_adapter()
