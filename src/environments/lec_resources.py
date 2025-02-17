from dataclasses import dataclass, field
from typing import Optional, Dict

from src.resources import Storage, Gateway
from src.resources.common_battery import CommonBattery
from src.resources.load import Load
from src.resources.generator import Generator


@dataclass
class HouseholdResource:
    household_id: str
    load: Load
    generator: Optional[Generator] = None
    storage: Optional[Storage] = None

    def update_load(self, new_values):
        self.load.value = new_values


@dataclass
class EnvironmentResources:
    gateway: Gateway
    common_battery: CommonBattery
    households: Dict[str, HouseholdResource] = field(default_factory=dict)

    def add_household(self, household_resource: HouseholdResource, household_number: int):
        self.households[str(household_resource.household_id) + "_" + str(household_number)] = household_resource
