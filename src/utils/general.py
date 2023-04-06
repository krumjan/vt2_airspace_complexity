from pathlib import Path
import math
from typing import Tuple


def get_project_root() -> str:
    """
    Returns the root path of the project.

    Returns
    -------
    str
        root path of the project
    """
    return str(Path(__file__).parent.parent.parent)
