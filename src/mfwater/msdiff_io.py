"""Reading ``msdiff_out.csv``, shared by the paper-I and the MFDA workflows.

Kept at package level so that ``algo_chemical_model`` and ``algo_mfda`` can both
import it without a circular import.
"""

from __future__ import annotations

import math
from pathlib import Path


def parse_msdiff_d_raw(path: str | Path) -> float:
    """Read the diffusion coefficient (raw msdiff unit) from ``msdiff_out.csv``.

    Two formats exist. Older msdiff (paper I) writes
    ``D / 10^-12 m^2/s, delta_D / ..., K / ...`` with D in column 0 and no species
    column; current msdiff writes ``Species, D_0 / 10^-12 m^2/s, ...``. The column
    is located by header (``D_0 /`` or ``D /``; not ``D_z`` or ``delta_D``), the
    unit is checked, and the value of the last row is returned. The Hummer term
    K is a separate column and is never added.

    Parameters
    ----------
    path : str | Path
        ``msdiff_out.csv``.

    Returns
    -------
    float
        Diffusion coefficient D_0 in 1e-12 m^2/s (as written by msdiff).

    Raises
    ------
    RuntimeError
        If the file is malformed, has an unexpected unit, or D is not finite.
    """
    path = Path(path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"{path} contains no data rows.")
    header = [h.strip() for h in lines[0].split(",")]
    cols = [i for i, h in enumerate(header) if h.startswith(("D_0 /", "D /"))]
    if len(cols) != 1 or "10^-12 m^2/s" not in header[cols[0]]:
        raise RuntimeError(
            f"Unexpected msdiff header in {path}: {header}. Expected a column "
            "'D_0 / 10^-12 m^2/s' or 'D / 10^-12 m^2/s'."
        )
    value = float(lines[-1].split(",")[cols[0]].strip())
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite diffusion coefficient in {path}.")
    return value
