from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class Request:
    """Una línea de pedido: recoger en un productor y entregar en un punto de entrega."""
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
        self.duration = duration      # dict {(origen, destino): minutos}
        self.distance = distance      # dict {(origen, destino): km}
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

        requests = []
        for r in t["order_items"].itertuples():
            requests.append(Request(
                item_id=r.order_item_id,
                order_id=r.order_id,
                pickup_loc=loc_id[("producer", prod_of[r.product_id])],
                delivery_loc=loc_id[("delivery", deliv_of[r.order_id])],
                weight_kg=r.weight_kg,
                pickup_start=pd.Timestamp(r.availability_start),
                pickup_end=pd.Timestamp(r.availability_end),
                delivery_start=pd.Timestamp(r.delivery_start),
                delivery_end=pd.Timestamp(r.delivery_end),
            ))

        vehicles = [Vehicle(v.vehicle_id, v.capacity_kg, bool(v.refrigerated), v.avg_speed_kmh)
                    for v in t["vehicles"].itertuples()]

        dm = t["distance_matrix"]
        keys = zip(dm.origin_location_id, dm.destination_location_id)
        duration = dict(zip(keys, dm.duration_min))
        keys = zip(dm.origin_location_id, dm.destination_location_id)
        distance = dict(zip(keys, dm.distance_km))

        depot = loc_id[("center", t["collection_centers"]["center_id"].iloc[0])]
        return cls(requests, vehicles, depot, duration, distance, rules)