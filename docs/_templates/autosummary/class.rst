{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members: None, Callback, LightningCLI, LightningDataModule, LightningModule, Module, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, PushToHubMixin, SaveConfigCallback, SpecialTokensMixin, Tensor, Trainer, WandbLogger

   {% block methods %}
   {%- if '__init__' not in inherited_members %}
   .. automethod:: __init__
   {%- endif %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
