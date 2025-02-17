from dataclasses import dataclass
import numpy as np
@dataclass
class CommonBattery:
    soc: np.ndarray
    discharge_vals: np.ndarray
    charge_vals: np.ndarray
    capacity: float
    charge_efficiency: float
    max_charge_power: float
    max_discharge_power: float

    def charge(self, amount: float) -> float:
        available_space = 1.0 - self.soc
        charge_percentage = min(amount * self.max_charge_power * self.charge_efficiency, available_space)
        self.soc += charge_percentage
        charge_amount = charge_percentage * self.capacity
        return charge_amount

    def discharge(self, amount: float, current_timestep) -> float:
        discharge_percentage = min(amount * self.max_discharge_power, self.soc[current_timestep])
        self.soc -= discharge_percentage
        discharge_amount = discharge_percentage * self.capacity
        return discharge_amount

    def get_state_of_charge(self, current_timestep) -> float:
        return self.soc[current_timestep]