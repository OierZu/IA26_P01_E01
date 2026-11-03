from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Request:
    item_id: int
    order_id: int
    pickup_loc: int
    delivery_loc: int
    weight_kg: float
    pickup_start: pd.Timestamp
    pickup_end: pd.Timestamp
    delivery_start: pd.Timestamp
    delivery_end: pd.Timestamp

@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    capacity_kg: float
    refrigerated: bool
    speed_kmh: float

def _kind(location_type: str) -> str:
    t = location_type.lower()
    if "producer" in t:
        return "producer"
    if "center" in t or "warehouse" in t:
        return "center"
    return "delivery"

class Problem:
    def __init__(self, requests, vehicles, depot_loc, duration, distance, rules):
        self.requests = requests
        self.vehicles = vehicles
        self.depot_loc = depot_loc
        self.duration = duration
        self.distance = distance
        self.rules = rules

    @classmethod
    def from_csv(cls, folder):
        # TODO: Implement CSV loading and data extraction logic
        pass