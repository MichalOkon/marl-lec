from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Generator:
    production: np.ndarray
    upper_bound: np.ndarray
    generation_prediction: Optional[np.ndarray] = None