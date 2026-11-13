from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set
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

@dataclass(frozen=True)
class Stop:
    location_id: int
    action: str  # "pickup", "unload", "load", or "deliver"
    items: Tuple[int, ...] = field(default_factory=tuple)

@dataclass
class State:
    # Maps vehicle_id to an ordered sequence of Stops
    routes: Dict[int, List[Stop]]

def _kind(location_type: str) -> str:
    t = location_type.lower()
    if "producer" in t:
        return "producer"
    if "center" in t or "warehouse" in t:
        return "center"
    return "delivery"

class Problem:
    def __init__(self, requests: List[Request], vehicles: List[Vehicle], depot_loc: int, duration: dict, distance: dict, rules: dict):
        self.requests = requests
        self.req_map: Dict[int, Request] = {r.item_id: r for r in requests}
        self.vehicles = vehicles
        self.veh_map: Dict[int, Vehicle] = {v.vehicle_id: v for v in vehicles}
        self.depot_loc = depot_loc
        self.duration = duration
        self.distance = distance
        self.rules = rules

    @classmethod
    def from_csv(cls, folder):
        folder = Path(folder)
        t = {f.stem: pd.read_csv(f) for f in folder.glob("*.csv")}
        rules = t["rules"].set_index("rule")["value"].to_dict()

        loc = t["locations"].copy()
        loc["kind"] = loc["location_type"].map(_kind)
        loc_id = {(r.kind, r.ref_id): r.location_id for r in loc.itertuples()}

        prod_of = t["products"].set_index("product_id")["producer_id"]
        deliv_of = t["orders"].set_index("order_id")["delivery_location_id"]

        requests = [
            Request(
                item_id=r.order_item_id,
                order_id=r.order_id,
                pickup_loc=loc_id[("producer", prod_of[r.product_id])],
                delivery_loc=loc_id[("delivery", deliv_of[r.order_id])],
                weight_kg=r.weight_kg,
                pickup_start=pd.Timestamp(r.availability_start),
                pickup_end=pd.Timestamp(r.availability_end),
                delivery_start=pd.Timestamp(r.delivery_start),
                delivery_end=pd.Timestamp(r.delivery_end),
            )
            for r in t["order_items"].itertuples()
        ]

        vehicles = [
            Vehicle(v.vehicle_id, v.capacity_kg, bool(v.refrigerated), v.avg_speed_kmh)
            for v in t["vehicles"].itertuples()
        ]

        dm = t["distance_matrix"]
        duration = dict(zip(zip(dm.origin_location_id, dm.destination_location_id), dm.duration_min))
        distance = dict(zip(zip(dm.origin_location_id, dm.destination_location_id), dm.distance_km))

        depot = loc_id[("center", t["collection_centers"]["center_id"].iloc[0])]
        return cls(requests, vehicles, depot, duration, distance, rules)

    def calculate_cost(self, state: State) -> float:
        total_distance = 0.0
        for route in state.routes.values():
            if not route:
                continue
            current_loc = self.depot_loc
            for stop in route:
                if current_loc != stop.location_id:
                    total_distance += self.distance.get((current_loc, stop.location_id), float("inf"))
                current_loc = stop.location_id
            if current_loc != self.depot_loc:
                total_distance += self.distance.get((current_loc, self.depot_loc), float("inf"))
        return total_distance

    def is_feasible(self, state: State, check_capacity: bool = False, check_complete: bool = True) -> bool:
        """
        Iteration 1: check_capacity=False (validates producer -> warehouse -> delivery flow).
        Iteration 2: check_capacity=True  (validates vehicle weight limits).
        """
        picked_up: Set[int] = set()
        at_warehouse: Set[int] = set()
        delivered: Set[int] = set()

        # Process all pickups and warehouse unloads, plus per-vehicle sequence/capacity
        for v_id, route in state.routes.items():
            if v_id not in self.veh_map:
                return False
            vehicle = self.veh_map[v_id]
            in_truck: Set[int] = set()

            for stop in route:
                if stop.action == "pickup":
                    for item_id in stop.items:
                        req = self.req_map.get(item_id)
                        if not req or stop.location_id != req.pickup_loc or item_id in picked_up:
                            return False
                        picked_up.add(item_id)
                        in_truck.add(item_id)

                elif stop.action == "unload":
                    if stop.location_id != self.depot_loc:
                        return False
                    for item_id in stop.items:
                        if item_id not in in_truck:
                            return False
                        in_truck.remove(item_id)
                        at_warehouse.add(item_id)

                elif stop.action == "load":
                    if stop.location_id != self.depot_loc:
                        return False
                    for item_id in stop.items:
                        if item_id in in_truck:
                            return False
                        in_truck.add(item_id)

                elif stop.action == "deliver":
                    for item_id in stop.items:
                        req = self.req_map.get(item_id)
                        if not req or stop.location_id != req.delivery_loc or item_id not in in_truck:
                            return False
                        in_truck.remove(item_id)
                        delivered.add(item_id)
                else:
                    return False

                if check_capacity:
                    current_weight = sum(self.req_map[i].weight_kg for i in in_truck)
                    if current_weight > vehicle.capacity_kg:
                        return False

            # Truck must be empty at the end of its route
            if in_truck:
                return False

        # Verifying every loaded item actually reached the warehouse first
        for v_id, route in state.routes.items():
            for stop in route:
                if stop.action == "load":
                    for item_id in stop.items:
                        if item_id not in at_warehouse:
                            return False
                        at_warehouse.remove(item_id)

        if check_complete and len(delivered) != len(self.requests):
            return False

        return True