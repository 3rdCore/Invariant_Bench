from __future__ import annotations

"""Utility helpers for saving matplotlib figures into the project-level 'plots/' folder.

Typical usage:
    from subpopbench.utils.plotting import savefig
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
    # This file lives at <root>/subpopbench/utils/plotting.py -> parents[2] == <root>
    return Path(__file__).resolve().parents[2]


def plots_dir() -> Path:
    """Return the Path to the top-level plots directory, creating it if missing."""
    p = _project_root() / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _coerce_figure(fig_or_ax: Optional[FigureLike]):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes  # type: ignore
    from matplotlib.figure import Figure  # type: ignore

    if fig_or_ax is None:
        return plt.gcf()
    if isinstance(fig_or_ax, Figure):
        return fig_or_ax
    if isinstance(fig_or_ax, Axes):
        return fig_or_ax.get_figure()
    raise TypeError("Expected Figure, Axes, or None; got %r" % (type(fig_or_ax),))


def savefig(
    filename: str,
    fig: Optional[FigureLike] = None,
    subdir: Optional[str] = None,
    dpi: int = 300,
    tight: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    **savefig_kwargs,
) -> Path:
    """Save a matplotlib figure into plots/ (optionally a subdir) and return its path.

    Parameters
    ----------
    filename : str
        Target file name. Extension determines format (e.g. .png, .pdf, .svg).
    fig : Figure | Axes | None
        Figure or Axes to save. If None, uses current active figure.
    subdir : str | None
        Optional subfolder inside plots/.
    dpi : int
        Resolution.
    tight : bool
        Apply bbox_inches='tight' to reduce whitespace.
    overwrite : bool
        If False and file exists, append an incrementing suffix before the extension.
    verbose : bool
        Print resulting path when saved.
    **savefig_kwargs : dict
        Forwarded to matplotlib Figure.savefig.
    """
    base = plots_dir()
    if subdir:
        base = base / subdir
        base.mkdir(parents=True, exist_ok=True)

    # Ensure extension exists; default to .png
    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(".png")

    dest = base / path.name

    if not overwrite:
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = base / f"{stem}_{counter}{suffix}"
            counter += 1

    fig_obj = _coerce_figure(fig)

    save_args = dict(dpi=dpi, **savefig_kwargs)
    if tight:
        save_args.setdefault("bbox_inches", "tight")
    fig_obj.savefig(dest, **save_args)
    if verbose:
        print(f"[savefig] Saved figure to {dest.relative_to(_project_root())}")
    return dest


__all__ = ["savefig", "plots_dir"]
