"""segvigen — 3D mesh segmentation using SegviGen models.

Three segmentation modules
--------------------------
- :mod:`segvigen.interactive`   — click-point conditioned segmentation
- :mod:`segvigen.full`          — full segmentation (image-conditioned)
- :mod:`segvigen.full_guided`   — full segmentation conditioned on a 2D guidance map

Supporting modules
------------------
- :mod:`segvigen.presets`  — sampler preset dictionaries

Quick-start
-----------
>>> import segvigen
>>> segvigen.interactive.run(glb_path=..., ckpt_path=..., ...)
>>> segvigen.full.run(glb_path=..., ckpt_path=..., ...)
>>> segvigen.full_guided.run(glb_path=..., ckpt_path=..., ...)
"""

from segvigen import interactive, full, full_guided
from segvigen.presets import SAMPLER_PRESETS

__all__ = [
    "interactive",
    "full",
    "full_guided",
    "SAMPLER_PRESETS",
]
