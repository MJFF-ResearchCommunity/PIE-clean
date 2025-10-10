import logging
import sys

from .constants import *
from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor

logger = logging.getLogger("PIE")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
