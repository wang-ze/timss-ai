"""Moved to ``timss_math.config``, which both subpackages share. This name stays the same module, so that older imports,
such as ``from timss_math.item_parameter_prediction.config import ...``, keep working."""

import sys

from .. import config

sys.modules[__name__] = config
