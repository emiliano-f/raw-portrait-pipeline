"""Built-in :class:`~raw_pipeline.core.protocols.Retoucher` implementations.

Importing this package registers ``none`` and ``frequency_separation``.
"""

from raw_pipeline.stages.retouchers.frequency_separation import FrequencySeparationRetoucher
from raw_pipeline.stages.retouchers.null import NullRetoucher

__all__ = ["FrequencySeparationRetoucher", "NullRetoucher"]
