"""Backward-compatible import path of :mod:`mfwater.md.pipeline`.

The implementation moved to the shared MD layer; everything is re-exported here.
``shutil`` and ``subprocess`` are imported because tests patch
``mfwater.algo_mfda.md_pipeline.shutil.which`` and ``...subprocess.run`` (the same
module objects that :mod:`mfwater.md.pipeline` uses).
"""

import shutil  # noqa: F401
import subprocess  # noqa: F401

from ..md.pipeline import (  # noqa: F401
    COMBINED_TRAJECTORY,
    DEFAULT_LAMMPS_CMD,
    MSDIFF_TO_NANO,
    PAPER_I_MSDIFF_ARGS,
    RESUME_HEADER,
    TEMPLATE_DIR,
    TRAVIS_DEPTH_PROMPT,
    TRAVIS_FRAMES_PROMPT,
    TRAVIS_MSD_FILE,
    TRAVIS_MSD_FILE_XY,
    TRAVIS_MSD_FILE_Z,
    MDForwardModel,
    _run,
    _set_answers,
    find_restart_files,
    find_travis_msd_file,
    iter_frames,
    merge_trajectories,
    parse_msdiff_output,
    patch_travis_input,
    select_restart_file,
    write_lammps_input,
    write_resume_input,
)
from ..md.protocol import MFDA_MSD_DEPTH_FRACTION as MSD_DEPTH_FRACTION  # noqa: F401
from ..md.protocol import (
    MFDA_PROTOCOL,
)
from ..md.system import calc_box_size, calc_cpus  # noqa: F401

# Former module constants of the MFDA pipeline; they are the values of the
# ``mfda`` protocol now and are no longer used by the logic.
EQ_END_STEP = MFDA_PROTOCOL.eq_end_step
PROD_END_STEP = MFDA_PROTOCOL.prod_end_step
DUMP_EVERY = MFDA_PROTOCOL.dump_every
RESTART_EVERY = MFDA_PROTOCOL.restart_every
MFDA_INPUT_TEMPLATE = MFDA_PROTOCOL.lammps_template
PRODUCTION_FRAMES = (PROD_END_STEP - EQ_END_STEP) // DUMP_EVERY
MSD_CORRELATION_DEPTH = MFDA_PROTOCOL.travis_depth
