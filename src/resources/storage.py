from dataclasses import dataclass, field
from typing import Union

import numpy as np

# Local storage (or local battery)
@dataclass
class Storage:
    soc: Union[np.ndarray, float]
    capacity_max: float
    initial_charge: float
    discharge_efficiency: float
    discharge_max: float
    charge_efficiency: float
    charge_max: float
    discharge: Union[np.ndarray, float]
    charge: Union[np.ndarray, float]

