"""
Helpers to read the model groups of a legacy (paper-I) HDF5 file in a defined order.

The legacy algorithms address models by the group names ``model_1``, ``model_2``, ...
h5py iterates over the names alphabetically, so ``model_10`` comes before ``model_2``.
Relying on that order is wrong for ten or more models, and the MFMC mathematics
(:mod:`mfwater.mfmc_core`) requires an explicit order anyway. All legacy algorithms
therefore obtain their models through the functions of this module.
"""

from __future__ import annotations

import re

import h5py
import numpy as np
import numpy.typing as npt

_NAME_RE = re.compile(r"^model_(\d+)$")


def _name_number(name: str) -> int:
    """
    Return the integer ``k`` of a group name ``model_k``.

    Parameters
    ----------
    name : str
        Group name.

    Returns
    -------
    int
        The number ``k``; names that do not match ``model_<k>`` give a number larger
        than any real one, so that they sort last.
    """
    m = _NAME_RE.match(name)
    return int(m.group(1)) if m else np.iinfo(np.int64).max


def model_groups(models: h5py.Group) -> list[tuple[str, h5py.Group]]:
    """
    List the model groups of the ``models`` group, ordered by decreasing size.

    The high-fidelity model is the one with the most molecules, so the first entry is
    the high-fidelity model. Models with the same ``n_molecules`` are ordered by the
    number in their name.

    Parameters
    ----------
    models : h5py.Group
        The ``models`` group of an input file.

    Returns
    -------
    list of tuple of (str, h5py.Group)
        Name and group of every model, ordered by decreasing ``n_molecules``.

    Raises
    ------
    RuntimeError
        If the file contains no model group or a model lacks ``n_molecules``.
    """
    items = [(n, g) for n, g in models.items() if isinstance(g, h5py.Group)]
    if not items:
        raise RuntimeError("The input file contains no model groups.")
    for name, grp in items:
        if "n_molecules" not in grp.attrs:
            raise RuntimeError(f"Model {name} has no attribute 'n_molecules'.")
    return sorted(
        items, key=lambda it: (-int(it[1].attrs["n_molecules"]), _name_number(it[0]))
    )


def models_by_rho(models: h5py.Group) -> list[tuple[str, h5py.Group]]:
    """
    List the model groups ordered by decreasing squared correlation coefficient.

    This is the ordering required by the allocation, the control-variate
    coefficients and the RMSE (:mod:`mfwater.mfmc_core`). Ties in ``rho**2`` are
    broken by decreasing ``n_molecules`` and then by the number in the group name.

    Parameters
    ----------
    models : h5py.Group
        The ``models`` group of an input file whose models have the attribute
        ``correlation``.

    Returns
    -------
    list of tuple of (str, h5py.Group)
        Name and group of every model, ordered by decreasing ``correlation**2``.

    Raises
    ------
    RuntimeError
        If a model lacks the attribute ``correlation`` (or ``n_molecules``).
    """
    items = model_groups(models)
    for name, grp in items:
        if "correlation" not in grp.attrs:
            raise RuntimeError(f"Model {name} has no attribute 'correlation'.")
    return sorted(
        items,
        key=lambda it: (
            -float(it[1].attrs["correlation"]) ** 2,
            -int(it[1].attrs["n_molecules"]),
            _name_number(it[0]),
        ),
    )


def attr_array(
    items: list[tuple[str, h5py.Group]], attr: str
) -> npt.NDArray[np.float64]:
    """
    Collect one attribute of all given models into a float64 array.

    Parameters
    ----------
    items : list of tuple of (str, h5py.Group)
        Models, e.g. from :func:`model_groups`; the result keeps this order.
    attr : str
        Attribute name.

    Returns
    -------
    numpy.ndarray
        The attribute values as ``float64``, one per model.

    Raises
    ------
    RuntimeError
        If a model lacks the attribute.
    """
    values = []
    for name, grp in items:
        if attr not in grp.attrs:
            raise RuntimeError(f"Model {name} has no attribute '{attr}'.")
        values.append(float(grp.attrs[attr]))
    return np.array(values, dtype=np.float64)
