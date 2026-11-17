"""
src/model.py
Domain model: Request, Vehicle, Stop, State, Problem (cost + feasibility checks).

Time is measured in minutes since the start of the planning horizon
(00:00 of rules.start_date). Vehicles start at the warehouse (depot) and must end there.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd


@dataclass(frozen=True)
class Request:
    """One order item: collect at a producer, deliver at a delivery location."""
    item_id: int
    order_id: int
    pickup_loc: int
    delivery_loc: int
    weight_kg: float
    pickup_start: pd.Timestamp
    pickup_end: pd.Timestamp
    delivery_start: pd.Timestamp
    delivery_end: pd.Timestamp
    category: str = ""


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    capacity_kg: float
    refrigerated: bool
    speed_kmh: float


@dataclass(frozen=True)
class Stop:
    location_id: int
    action: str  # "pickup", "unload", "load", "deliver" or "return"
    items: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class State:
    # Maps vehicle_id to an ordered sequence of Stops (all trips of the vehicle, one after another)
    routes: Dict[int, List[Stop]]


def _kind(location_type: str) -> str:
    t = location_type.lower()
    if "producer" in t:
        return "producer"
    if "center" in t or "warehouse" in t:
        return "center"
    return "delivery"


class Problem:
    def __init__(self, requests: List[Request], vehicles: List[Vehicle], depot_loc: int,
                 duration: dict, distance: dict, rules: dict):
        self.requests = requests
        self.req_map: Dict[int, Request] = {r.item_id: r for r in requests}
        self.vehicles = vehicles
        self.veh_map: Dict[int, Vehicle] = {v.vehicle_id: v for v in vehicles}
        self.depot_loc = depot_loc
        self.duration = duration
        self.distance = distance
        self.rules = rules

        # numeric rules (rules.csv)
        self.service_min = float(rules.get("stop_service_min", 10))
        self.handling_min = float(rules.get("warehouse_handling_min", 15))
        self.max_dwell_min = float(rules.get("max_warehouse_dwell_h", 24)) * 60
        self.start = pd.Timestamp(rules.get("start_date", min(r.pickup_start for r in requests).date()))

        # items grouped by order
        self.order_items: Dict[int, List[int]] = {}
        for r in requests:
            self.order_items.setdefault(r.order_id, []).append(r.item_id)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_csv(cls, folder):
        folder = Path(folder)
        t = {f.stem: pd.read_csv(f) for f in folder.glob("*.csv")}
        rules = t["rules"].set_index("rule")["value"].to_dict()

        # locations.location_id is the routing id used by the distance matrix
        loc = t["locations"].copy()
        loc["kind"] = loc["location_type"].map(_kind)
        loc_id = {(r.kind, r.ref_id): r.location_id for r in loc.itertuples()}

        prod_of = t["products"].set_index("product_id")["producer_id"]
        cat_of = t["products"].set_index("product_id")["category"]
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
                category=cat_of[r.product_id],
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

    # ------------------------------------------------------------------ small helpers
    def minutes(self, ts) -> float:
        """Timestamp -> minutes since the start of the horizon."""
        return (ts - self.start).total_seconds() / 60.0

    def clock(self, minutes: float) -> str:
        """Minutes since the start -> 'dd HH:MM' text."""
        ts = self.start + pd.Timedelta(minutes=minutes)
        return ts.strftime("%d %H:%M")

    def km(self, a: int, b: int) -> float:
        return 0.0 if a == b else self.distance.get((a, b), float("inf"))

    def travel_min(self, a: int, b: int) -> float:
        return 0.0 if a == b else self.duration.get((a, b), float("inf"))

    def order_loc(self, order_id: int) -> int:
        return self.req_map[self.order_items[order_id][0]].delivery_loc

    def order_weight(self, order_id: int) -> float:
        return sum(self.req_map[i].weight_kg for i in self.order_items[order_id])

    def order_pickup_day(self, order_id: int):
        """Day on which the order is collected (all its items share the same day in the datasets)."""
        return self.req_map[self.order_items[order_id][0]].pickup_start.date()

    def order_delivery_day(self, order_id: int):
        """Day on which the order must be delivered (it can be later than the pickup day)."""
        return self.req_map[self.order_items[order_id][0]].delivery_start.date()

    def items_at(self, day, producer_loc) -> List[int]:
        """Items collected at a producer on a given day (they are picked up in a single visit)."""
        return [r.item_id for r in self.requests
                if r.pickup_loc == producer_loc and r.pickup_start.date() == day]

    def pickup_window(self, item_ids) -> Tuple[float, float]:
        """A single visit must start inside every item window: intersect them."""
        reqs = [self.req_map[i] for i in item_ids]
        return (max(self.minutes(r.pickup_start) for r in reqs),
                min(self.minutes(r.pickup_end) for r in reqs))

    def delivery_window(self, item_ids) -> Tuple[float, float]:
        reqs = [self.req_map[i] for i in item_ids]
        return (max(self.minutes(r.delivery_start) for r in reqs),
                min(self.minutes(r.delivery_end) for r in reqs))

    # ------------------------------------------------------------------ plan -> routes
    def plan_to_state(self, plan) -> State:
        """
        plan = {day: {"collect": {vehicle_id: [producer_loc, ...]},
                      "deliver": {vehicle_id: [order_id, ...]}}}   (lists are visit orders)
        Each day a vehicle makes up to three legs, in this order:
          1. delivery of orders already in the warehouse (collected on an earlier day)
          2. collection leg: producers -> warehouse (unload)
          3. delivery of the orders collected today
        Doing 1 before 2 keeps the warehouse dwell time (24 h) under control.
        """
        routes = {v.vehicle_id: [] for v in self.vehicles}
        for day in sorted(plan):
            for vid in routes:
                orders = plan[day]["deliver"].get(vid, [])
                old = [o for o in orders if self.order_pickup_day(o) < day]
                new = [o for o in orders if self.order_pickup_day(o) >= day]
                routes[vid] += self._delivery_leg(old)
                routes[vid] += self._collection_leg(day, plan[day]["collect"].get(vid, []))
                routes[vid] += self._delivery_leg(new)
        return State(routes=routes)

    def _collection_leg(self, day, producers) -> List[Stop]:
        if not producers:
            return []
        items_by_stop = [self.items_at(day, p) for p in producers]
        stops = [Stop(p, "pickup", tuple(ids)) for p, ids in zip(producers, items_by_stop)]
        stops.append(Stop(self.depot_loc, "unload", tuple(i for ids in items_by_stop for i in ids)))
        return stops

    def _delivery_leg(self, orders) -> List[Stop]:
        if not orders:
            return []
        all_items = tuple(i for o in orders for i in self.order_items[o])
        stops = [Stop(self.depot_loc, "load", all_items)]
        stops += [Stop(self.order_loc(o), "deliver", tuple(self.order_items[o])) for o in orders]
        stops.append(Stop(self.depot_loc, "return"))
        return stops

    # ------------------------------------------------------------------ cost
    def calculate_cost(self, state: State) -> float:
        """Total distance (km). Every vehicle starts and ends at the depot."""
        total_distance = 0.0
        for route in state.routes.values():
            if not route:
                continue
            current_loc = self.depot_loc
            for stop in route:
                total_distance += self.km(current_loc, stop.location_id)
                current_loc = stop.location_id
            total_distance += self.km(current_loc, self.depot_loc)
        return total_distance

    # ------------------------------------------------------------------ feasibility
    def is_feasible(self, state: State, check_capacity: bool = False, check_complete: bool = True) -> bool:
        """
        Spatial rules only (no clock): producer -> warehouse -> delivery flow and,
        optionally, vehicle capacity. Time windows are checked by check_time().
        """
        picked_up: Set[int] = set()
        at_warehouse: Set[int] = set()
        delivered: Set[int] = set()

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

                elif stop.action == "return":
                    # end of a trip: back at the depot with an empty truck
                    if stop.location_id != self.depot_loc or in_truck:
                        return False
                else:
                    return False

                if check_capacity:
                    current_weight = sum(self.req_map[i].weight_kg for i in in_truck)
                    if current_weight > vehicle.capacity_kg:
                        return False

            # Truck must be empty at the end of its route
            if in_truck:
                return False

        # Every loaded item must have reached the warehouse first
        for route in state.routes.values():
            for stop in route:
                if stop.action == "load":
                    for item_id in stop.items:
                        if item_id not in at_warehouse:
                            return False
                        at_warehouse.remove(item_id)

        if check_complete and len(delivered) != len(self.requests):
            return False

        return True

    def simulate(self, state: State):
        """
        Walk every route through time (waiting is allowed when a vehicle arrives early).
        A route is a sequence of legs: collection legs (pickups ... unload) and delivery legs
        (load ... return). Collection legs are simulated first so that a delivery leg can wait
        until its items have been unloaded by any vehicle.
        Returns (rows, problems, info): one row per stop, a list of violations,
        and totals (finish time per vehicle, total waiting minutes).
        """
        rows, problems = [], []
        arrive_wh, unload_end, load_start = {}, {}, {}
        clock = {v: 0.0 for v in state.routes}
        where = {v: self.depot_loc for v in state.routes}
        count = {v: 0 for v in state.routes}
        waited = [0.0]

        # split each route in legs: a leg ends at "unload" or "return"
        legs = {}
        for v, route in state.routes.items():
            legs[v], cur = [], []
            for stop in route:
                cur.append(stop)
                if stop.action in ("unload", "return"):
                    legs[v].append(cur)
                    cur = []
            if cur:
                legs[v].append(cur)

        def run_leg(v, leg):
            t, loc = clock[v], where[v]
            for stop in leg:
                count[v] += 1
                t += self.travel_min(loc, stop.location_id)
                loc = stop.location_id
                arrival, opens, closes = t, None, None

                if stop.action in ("pickup", "deliver"):
                    if stop.action == "pickup":
                        opens, closes = self.pickup_window(stop.items)
                    else:
                        opens, closes = self.delivery_window(stop.items)
                    if t < opens:
                        waited[0] += opens - t
                        t = opens
                    if t > closes + 1e-9:
                        problems.append(f"vehicle {v}, stop {count[v]} ({stop.action} at {loc}): "
                                        f"starts {self.clock(t)} after the window closes {self.clock(closes)}")
                    rows.append((v, count[v], stop.action, loc, arrival, t, opens, closes, stop.items))
                    t += self.service_min
                elif stop.action == "unload":
                    for i in stop.items:
                        arrive_wh[i] = t
                        unload_end[i] = t + self.handling_min
                    rows.append((v, count[v], stop.action, loc, arrival, t, None, None, stop.items))
                    t += self.handling_min
                elif stop.action == "load":
                    ready = [unload_end[i] for i in stop.items if i in unload_end]
                    if len(ready) < len(stop.items):
                        problems.append(f"vehicle {v}: loads items that were never unloaded")
                    if ready and t < max(ready):          # wait until every item is in the warehouse
                        waited[0] += max(ready) - t
                        t = max(ready)
                    for i in stop.items:
                        load_start[i] = t
                    rows.append((v, count[v], stop.action, loc, arrival, t, None, None, stop.items))
                    t += self.handling_min
                else:  # return
                    rows.append((v, count[v], stop.action, loc, arrival, t, None, None, stop.items))
            clock[v], where[v] = t, loc

        def runnable(leg):
            # a delivery leg can start only when all its items are already in the warehouse
            return leg[0].action != "load" or all(i in unload_end for i in leg[0].items)

        while any(legs.values()):
            pending = [v for v in legs if legs[v]]
            ready_now = [v for v in pending if runnable(legs[v][0])]
            collecting = [v for v in ready_now if legs[v][0][0].action == "pickup"]
            for v in (collecting or ready_now or pending):
                run_leg(v, legs[v].pop(0))

        finish = {}
        for v in state.routes:
            t = clock[v]
            if where[v] != self.depot_loc:   # a route that does not end at the depot drives back
                t += self.travel_min(where[v], self.depot_loc)
            finish[v] = t

        # warehouse dwell: arrival -> start of loading
        for i, t_load in load_start.items():
            stay = t_load - arrive_wh.get(i, t_load)
            if stay > self.max_dwell_min + 1e-9:
                problems.append(f"item {i} stays {stay / 60:.1f} h in the warehouse")

        return rows, problems, {"finish": finish, "wait_min": waited[0]}

    def check_time(self, state: State) -> bool:
        """True if every time window and the warehouse dwell rule are respected."""
        return len(self.simulate(state)[1]) == 0

    def is_feasible_full(self, state: State) -> bool:
        """Flow + capacity + time windows + dwell."""
        return self.is_feasible(state, check_capacity=True) and self.check_time(state)

    def route_table(self, state: State) -> pd.DataFrame:
        """Readable schedule of a solution (used in the notebooks)."""
        rows, _, _ = self.simulate(state)
        return pd.DataFrame([{
            "vehicle": v, "stop": n, "action": a, "location": loc,
            "arrival": self.clock(arr), "service_start": self.clock(st),
            "window": "" if o is None else f"{self.clock(o)} - {self.clock(c)}",
            "items": list(items),
        } for v, n, a, loc, arr, st, o, c, items in rows])
