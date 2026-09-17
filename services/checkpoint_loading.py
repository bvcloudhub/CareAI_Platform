"""
Checkpoint loading shared by the PyTorch imaging services.

torch.load unpickles, and unpickling can execute arbitrary code. Every
checkpoint is tried with weights_only=True first, which restricts loading to
tensors and plain containers; the permissive path runs only for files that
need it (for example ones that pickled numpy scalars), and callers record which
path ran. These are the project's own trained models — do not route downloaded
files through the permissive path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Union


def load_checkpoint(path: Union[str, Path]) -> Tuple[Any, str]:
    """Return (checkpoint, load_mode) where load_mode is 'weights_only' or 'full pickle'."""
    import torch

    target = str(path)
    try:
        return torch.load(target, map_location="cpu", weights_only=True), "weights_only"
    except Exception:
        return torch.load(target, map_location="cpu", weights_only=False), "full pickle"
