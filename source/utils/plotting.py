from __future__ import annotations

"""Utility helpers for saving matplotlib figures into the project-level 'plots/' folder.

Typical usage:
    from source.utils.plotting import savefig
    # ... create a plot ...
    savefig("accuracy_over_time.png")

Features:
- Ensures the top-level 'plots/' directory exists.
- Optional subdirectories: savefig("curve.png", subdir="experiment1") -> plots/experiment1/curve.png
- Auto filename disambiguation when overwrite=False (default): adds _1, _2, ... if file exists.
- Accepts either a matplotlib Figure, Axes, or uses the current active figure.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # Only for type checkers; avoids importing heavy deps at runtime
    from matplotlib.axes import Axes  # pragma: no cover
    from matplotlib.figure import Figure  # pragma: no cover

FigureLike = Union["Figure", "Axes"]  # type: ignore[name-defined]


def _project_root() -> Path:
    # This file lives at <root>/source/utils/plotting.py -> parents[2] == <root>
    return Path(__file__).resolve().parents[2]


def plots_dir() -> Path:
    """Return the Path to the top-level plots directory, creating it if missing."""
    p = _project_root() / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(filename: str, fig: Optional[FigureLike] = None, **savefig_kwargs):
    """
    Directly save a matplotlib figure using the native savefig function.
    Parameters:
        filename: str, path to save the figure (can include folders)
        fig: Figure or Axes or None (if None, uses current active figure)
        **savefig_kwargs: passed to matplotlib's savefig
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if fig is None:
        fig_obj = plt.gcf()
    elif isinstance(fig, Figure):
        fig_obj = fig
    elif isinstance(fig, Axes):
        fig_obj = fig.get_figure()
    else:
        raise TypeError(f"Expected Figure, Axes, or None; got {type(fig)}")
    fig_obj.savefig(filename, **savefig_kwargs)
    print(f"[savefig] Saved figure to {filename}")
    return filename


__all__ = ["savefig", "plots_dir"]
