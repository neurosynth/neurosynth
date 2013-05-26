# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""NeuroSynth -- large-scale synthesis of functional neuroimaging data.

"""
__all__ = ["analysis", "base", "set_logging_level", "__version__"]

import logging
import sys
import os

from version import __version__

logger = logging.getLogger("neurosynth")

def set_logging_level(level=None):
    """Set neurosynth's logging level

    Args
      level : str
        Name of the logging level (warning, error, info, etc) known
        to logging module.  If no level provided, it would get that one
        from environment variable NEUROSYNTH_LOGLEVEL
    """
    if level is None:
        level = os.environ.get('NEUROSYNTH_LOGLEVEL', 'warn')
    if level is not None:
        logger.setLevel(getattr(logging, level.upper()))
    return logger.getEffectiveLevel()

def _setup_logger(logger):
    # Basic logging setup
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)-6s %(module)-7s %(message)s"))
    logger.addHandler(console)
    set_logging_level()

# Setup neurosynth's logger
_setup_logger(logger)
