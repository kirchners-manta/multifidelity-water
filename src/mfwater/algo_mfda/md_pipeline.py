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
    DUMP_EVERY,
    EQ_END_STEP,
    MFDA_INPUT_TEMPLATE,
    MSD_CORRELATION_DEPTH,
    MSD_DEPTH_FRACTION,
    MSDIFF_TO_NANO,
    PAPER_I_MSDIFF_ARGS,
    PROD_END_STEP,
    PRODUCTION_FRAMES,
    RESTART_EVERY,
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
from ..md.system import calc_box_size, calc_cpus  # noqa: F401
