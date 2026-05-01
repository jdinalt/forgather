"""Deep One model converter (Forgather-only).

Deep One is a Deepnet-style residual-scaled transformer with ALiBi
attention. There is no HuggingFace counterpart — ``forgather convert``
is unsupported. ``forgather update`` works through the migration
chain registered here.
"""

import os
from typing import Any, Dict, override

from forgather import MetaConfig
from forgather.ml.model_conversion import ForgatherOnlyConverter, register_converter


@register_converter("deepone")
class DeeponeConverter(ForgatherOnlyConverter):
    """Forgather-only converter for Deep One models.

    Saved deepone models stamp ``forgather_arch = "deepone"`` (set in
    ``templatelib/examples/models/transformers/deepone.yaml``); this
    converter routes them on ``forgather update`` to the deepone
    project at ``examples/models/deepone``.
    """

    arch = "deepone"
    arch_version = "1"
    forgather_migrations: dict = {}

    def __init__(self):
        super().__init__(model_type="deepone")

    @override
    def get_project_info(self) -> Dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )
