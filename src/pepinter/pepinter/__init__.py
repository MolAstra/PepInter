from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure

define_import_structure = {
    "configuration_pepinter": ["PepInterConfig"],
    "modeling_pepinter": [
        "PepInterModel",
        "PepInterModelForMaskedLM",
        "PepInterModelForEnergy",
        "PepInterModelForClassification",
        "PepInterModelForAffinity",
    ],
}

if TYPE_CHECKING:
    from .configuration_pepinter import *
    from .modeling_pepinter import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__, _file, define_import_structure, module_spec=__spec__
    )
