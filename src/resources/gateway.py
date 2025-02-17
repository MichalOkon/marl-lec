from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Gateway:
    imports: np.ndarray
    exports: np.ndarray
    import_price: np.ndarray
    export_price: np.ndarray
    imported_cost: np.ndarray
    exported_cost: np.ndarray
    timestamps: np.ndarray
    price_prediction: Optional[np.ndarray] = None
