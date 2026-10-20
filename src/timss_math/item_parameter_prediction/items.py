"""Moved to ``timss_math.items``, which both subpackages share. This name stays the same module, so that older imports,
such as ``from timss_math.item_parameter_prediction.items import ...``, keep working."""

import sys

from .. import items

sys.modules[__name__] = items
