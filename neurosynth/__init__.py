"""NeuroSynth -- large-scale synthesis of functional neuroimaging data.

"""

__all__ = ["analysis", "base"]

import logging
import sys
import os

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
    console.setFormatter(logging.Formatter("%(levelname)-6s %(message)s"))
    logger.addHandler(console)
    set_logging_level()

# Setup neurosynth's logger
_setup_logger(logger)
