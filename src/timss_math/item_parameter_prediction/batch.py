"""Moved to ``timss_math.batch``, which both subpackages share. This name stays the same module, so that older imports,
such as ``from timss_math.item_parameter_prediction.batch import ...``, keep working."""

import sys

from .. import batch

sys.modules[__name__] = batch
