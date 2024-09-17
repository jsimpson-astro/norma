from .core import find_max
from .interactive import InteractiveNorma
from .utils import normalise

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)