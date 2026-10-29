"""TinyGestureSSD -- single-scale SSDLite hand-gesture detector (local training package).

Ported from the original TinyGestureSSD_train Colab notebook. Same math throughout
(anchor generation, matching, loss, post-processing) -- only the project layout,
device handling, and dataset caching changed for local, GPU-accelerated use.
"""

__version__ = "0.3.0"
