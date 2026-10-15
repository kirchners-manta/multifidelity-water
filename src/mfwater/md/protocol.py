"""MD protocols: run lengths, output intervals and analysis settings.

A protocol bundles everything that defines *what* one MD evaluation computes,
independent of the algorithm that asks for it. Two protocols are registered:

``mfda``
    The shortened protocol of the MFDA algorithm: 250 ps equilibration, 1 ns
    production, 1000 frames analysed with a correlation depth of 300 frames.
``paper1``
    The protocol of paper I (Froembgen et al., J. Chem. Phys. 164, 024504):
    1 ns equilibration, 2 ns production, TRAVIS correlation depth of 600 frames.
    The LAMMPS input is the paper-I template plus only ``restart`` and
    ``run ... upto`` (neither changes the dynamics), so that interrupted runs can
    be resumed.

The numbers in a protocol are the single source of truth for the resume input,
the trajectory merge and the TRAVIS input. The LAMMPS template contains the same
numbers as literal text; :meth:`MDProtocol.validate_template` checks that both
agree, so that a template edit cannot silently change a protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Directory with the LAMMPS templates of the protocols.
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Fraction of the analysed frames used as MSD correlation depth by MFDA. Fixed
#: convention (30 %): the long-lag end of an MSD is averaged over few windows
#: and noisy, so only the first 30 % of the trajectory length is used.
MFDA_MSD_DEPTH_FRACTION = 0.30


@dataclass(frozen=True)
class MDProtocol:
    """Definition of one MD evaluation protocol.

    All steps are *absolute* LAMMPS timesteps after ``reset_timestep 0`` (the
    timestep size is 1 fs in all templates).

    Parameters
    ----------
    name : str
        Registry name; part of the cache key of every protocol except ``mfda``.
    lammps_template : Path
        LAMMPS input template with the placeholders ``VAR_EPS``, ``VAR_SIG``,
        ``VAR_q_H`` and ``VAR_q_O``; uses ``restart`` and ``run ... upto``.
    eq_end_step : int
        Timestep at the end of the equilibration run.
    prod_end_step : int
        Timestep at the end of the production run.
    dump_every : int
        Interval of the trajectory dumps in steps (also the frame spacing of the
        TRAVIS analysis).
    restart_every : int
        Interval of the LAMMPS restart files in steps.
    minimize_max_iter : int
        Iteration cap of the energy minimisation.
    minimize_max_eval : int
        Force-evaluation cap of the energy minimisation.
    travis_depth : int | None
        MSD correlation depth in frames written into the TRAVIS input, or None
        to keep the answer of the TRAVIS template.
    travis_frames : int | None
        Number of trajectory frames TRAVIS reads, or None to keep the answer of
        the TRAVIS template (paper I: 2000 frames for cubic boxes, all frames
        for OrthoBoXY).
    """

    name: str
    lammps_template: Path
    eq_end_step: int
    prod_end_step: int
    dump_every: int
    restart_every: int
    minimize_max_iter: int
    minimize_max_eval: int
    travis_depth: int | None
    travis_frames: int | None

    def validate_template(self) -> None:
        """Check that the LAMMPS template agrees with the numbers of the protocol.

        Raises
        ------
        RuntimeError
            If the template lacks the ``minimize``, ``restart``, ``dump`` or
            ``run ... upto`` commands or any of them disagrees with the protocol.
        """
        cmds = [
            line.split("#")[0].split()
            for line in self.lammps_template.read_text(encoding="utf-8").splitlines()
        ]
        cmds = [c for c in cmds if c]
        runs = [c for c in cmds if c[0] == "run"]
        mins = [c for c in cmds if c[0] == "minimize"]
        restarts = [c for c in cmds if c[0] == "restart"]
        dumps = {c[1]: c for c in cmds if c[0] == "dump"}
        problems: list[str] = []
        if len(mins) != 1 or [int(x) for x in mins[0][3:5]] != [
            self.minimize_max_iter,
            self.minimize_max_eval,
        ]:
            problems.append(
                f"minimize caps {[m[3:5] for m in mins]} != "
                f"({self.minimize_max_iter}, {self.minimize_max_eval})"
            )
        if len(restarts) != 1 or restarts[0][1] != str(self.restart_every):
            problems.append(f"restart {restarts} != every {self.restart_every}")
        if [r[1:] for r in runs] != [
            [str(self.eq_end_step), "upto"],
            [str(self.prod_end_step), "upto"],
        ]:
            problems.append(
                f"run commands {[' '.join(r) for r in runs]} != "
                f"'run {self.eq_end_step} upto', 'run {self.prod_end_step} upto'"
            )
        for dump_id in ("dEQUI", "dPROD"):
            cmd = dumps.get(dump_id)
            if cmd is None or cmd[3:5] != ["custom", str(self.dump_every)]:
                problems.append(f"dump {dump_id}: {cmd} != custom {self.dump_every}")
        if problems:
            raise RuntimeError(
                f"LAMMPS template {self.lammps_template} contradicts protocol "
                f"{self.name!r}: " + "; ".join(problems)
            )


#: MFDA protocol: 250 ps + 1 ns. TRAVIS reads 1000 frames (one fewer than the
#: 1001 dumped, like the 2000 of 2001 frames of paper I) with depth 300 (30 %).
MFDA_PROTOCOL = MDProtocol(
    name="mfda",
    lammps_template=DATA_DIR / "input_mfda.lmp",
    eq_end_step=250_000,
    prod_end_step=1_250_000,
    dump_every=1000,
    restart_every=100_000,
    minimize_max_iter=10_000,
    minimize_max_eval=100_000,
    travis_frames=1000,
    travis_depth=round(MFDA_MSD_DEPTH_FRACTION * 1000),
)

#: Paper-I protocol: 1 ns + 2 ns (the paper-I ``run 2000000`` after the first run
#: ends at absolute step 3 000 000), caps 100/1000, depth 600; the frame count
#: stays that of the paper-I TRAVIS templates.
PAPER1_PROTOCOL = MDProtocol(
    name="paper1",
    lammps_template=DATA_DIR / "input_paper1.lmp",
    eq_end_step=1_000_000,
    prod_end_step=3_000_000,
    dump_every=1000,
    restart_every=100_000,
    minimize_max_iter=100,
    minimize_max_eval=1000,
    travis_frames=None,
    travis_depth=600,
)

#: Registered protocols by name.
PROTOCOLS: dict[str, MDProtocol] = {p.name: p for p in (MFDA_PROTOCOL, PAPER1_PROTOCOL)}


def get_protocol(protocol: str | MDProtocol) -> MDProtocol:
    """Resolve a protocol name or object.

    Parameters
    ----------
    protocol : str | MDProtocol
        Registered name (``"mfda"``, ``"paper1"``) or a protocol object. An
        object is returned unchanged; its name must not clash with a registered
        protocol of different content (the name is part of the cache key).

    Returns
    -------
    MDProtocol
        The protocol.

    Raises
    ------
    ValueError
        If the name is unknown or an object reuses a registered name with
        different settings.
    """
    if isinstance(protocol, MDProtocol):
        known = PROTOCOLS.get(protocol.name)
        if known is not None and known != protocol:
            raise ValueError(
                f"Protocol name {protocol.name!r} is registered with different "
                "settings; use a new name (it is part of the cache key)."
            )
        return protocol
    try:
        return PROTOCOLS[protocol]
    except KeyError:
        raise ValueError(
            f"Unknown MD protocol {protocol!r}; registered: {sorted(PROTOCOLS)}."
        ) from None
