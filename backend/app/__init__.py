"""MusicMosaic backend package initialization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Librosa lazily imports Numba-decorated modules. An explicit cache locator
# avoids import races and editable-install source locator failures when feature
# extraction starts in worker threads.
_numba_cache = Path(tempfile.gettempdir()) / "musicmosaic-numba"
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))
os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
