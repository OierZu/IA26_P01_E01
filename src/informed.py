"""
src/informed.py
Informed search (Greedy best-first and A*) for the pickup-and-delivery problem.

Formulation (one search per vehicle, one trip per vehicle)
----------------------------------------------------------
Trip   : depot -> producers -> warehouse (depot) -> customers -> back to depot
State  : (current location, producers still to visit, warehouse done?, customers still to visit)
Action : go to one of the locations allowed in the current phase
           phase 1: any producer still to visit  (pickup)
           phase 2: the warehouse                (unload + load)
           phase 3: any customer still to visit  (deliver)
           phase 4: back to the depot            (return)
Cost   : distance in km between the two locations (prob.distance)
Goal   : everything visited and the vehicle is back at the depot

Heuristics (both admissible within this action space, so A* returns the optimum)
    h1: distance from the current location to the nearest location still required
    h2: sum, over every location still required, of the cheapest edge that can enter it

Optional time windows (use_time=True): arriving early means waiting, arriving late prunes the branch.
"""
import heapq
import itertools
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.model import Stop, State

INF = float("inf")


def _d(prob, a, b):
    """Distance in km (0 if same place)."""
    return 0.0 if a == b else prob.distance.get((a, b), INF)


def split_items(prob) -> Dict[int, List[int]]:
    """Greedy first-fit split of the items between vehicles by capacity (same idea as notebook 03)."""
    vehicles = sorted(prob.vehicles, key=lambda v: v.vehicle_id)
    assign = {v.vehicle_id: [] for v in vehicles}
    load = {v.vehicle_id: 0.0 for v in vehicles}
    for r in prob.requests:
        for v in vehicles:
            if load[v.vehicle_id] + r.weight_kg <= v.capacity_kg:
                assign[v.vehicle_id].append(r.item_id)
                load[v.vehicle_id] += r.weight_kg
                break
        else:
            raise ValueError(f"Item {r.item_id} does not fit in any vehicle")
    return assign


# --------------------------------------------------------------------------- heuristics
def _required(loc, depot, pend):
    """Locations that must still be visited (the depot is needed again if the vehicle is away or has stops left)."""
    req = set(pend)
    if pend or loc != depot:
        req.add(depot)
    return req


def make_heuristic(name, prob, nodes):
    depot = prob.depot_loc
    nodes = set(nodes) | {depot}
    # cheapest edge entering each node (used by h2)
    min_in = {s: min((_d(prob, u, s) for u in nodes if u != s), default=0.0) for s in nodes}

    if name == "h1":
        def h(loc, pend):
            req = _required(loc, depot, pend)
            return min((_d(prob, loc, s) for s in req if s != loc), default=0.0)
    elif name == "h2":
        def h(loc, pend):
            return sum(min_in[s] for s in _required(loc, depot, pend))
    else:
        raise ValueError("heuristic must be 'h1' or 'h2'")
    return h


# --------------------------------------------------------------------------- search
@dataclass
class SearchResult:
    vehicle_id: int
    algorithm: str
    heuristic: str
    route: List[Stop]
    cost: float
    expanded: int
    generated: int
    seconds: float
    status: str  # "solved", "no_solution" or "aborted"


def _build_route(path, pick_items, del_items, item_ids, depot):
    all_ids = tuple(item_ids)
    route = []
    for kind, loc in path:
        if kind == "pickup":
            route.append(Stop(loc, "pickup", tuple(pick_items[loc])))
        elif kind == "warehouse":
            route.append(Stop(depot, "unload", all_ids))
            route.append(Stop(depot, "load", all_ids))
        elif kind == "deliver":
            route.append(Stop(loc, "deliver", tuple(del_items[loc])))
        # "return" needs no Stop: calculate_cost already adds the way back to the depot
    return route


def solve_vehicle(prob, vehicle_id, item_ids, algorithm="astar", heuristic="h1",
                  use_time=False, start_time=None, max_expanded=500_000) -> SearchResult:
    """algorithm: 'greedy' (order by h) or 'astar' (order by g + h)."""
    if algorithm not in ("greedy", "astar"):
        raise ValueError("algorithm must be 'greedy' or 'astar'")
    astar = algorithm == "astar"
    depot = prob.depot_loc

    if not item_ids:
        return SearchResult(vehicle_id, algorithm, heuristic, [], 0.0, 0, 0, 0.0, "solved")

    pick_items, del_items = {}, {}
    for i in item_ids:
        r = prob.req_map[i]
        pick_items.setdefault(r.pickup_loc, []).append(i)
        del_items.setdefault(r.delivery_loc, []).append(i)

    h = make_heuristic(heuristic, prob, set(pick_items) | set(del_items) | {depot})

    # time windows in minutes since start_time
    if use_time:
        if start_time is None:
            start_time = min(r.pickup_start for r in prob.requests) - pd.Timedelta(hours=2)

        def mins(ts):
            return (ts - start_time).total_seconds() / 60.0

        pick_win = {loc: (max(mins(prob.req_map[i].pickup_start) for i in ids),
                          min(mins(prob.req_map[i].pickup_end) for i in ids))
                    for loc, ids in pick_items.items()}
        del_win = {loc: (max(mins(prob.req_map[i].delivery_start) for i in ids),
                         min(mins(prob.req_map[i].delivery_end) for i in ids))
                   for loc, ids in del_items.items()}

    def successors(loc, pp, wh, pdl):
        if pp:
            return [("pickup", p) for p in pp]
        if not wh:
            return [("warehouse", depot)]
        if pdl:
            return [("deliver", c) for c in pdl]
        if loc != depot:
            return [("return", depot)]
        return []

    counter = itertools.count()
    start = (depot, frozenset(pick_items), False, frozenset(del_items))
    h_start = h(depot, start[1] | start[3])
    heap = [(h_start, next(counter), 0.0, 0.0, start, ())]
    seen = {}  # state -> list of (g, t) already expanded (a node is skipped only if dominated)
    expanded = generated = 0
    status, best = "no_solution", None
    t_begin = time.perf_counter()

    while heap:
        _, _, g, t, state, path = heapq.heappop(heap)
        if any(g2 <= g and t2 <= t for g2, t2 in seen.get(state, ())):
            continue
        seen.setdefault(state, []).append((g, t))
        loc, pp, wh, pdl = state

        if not pp and wh and not pdl and loc == depot:
            best, status = (g, path), "solved"
            break

        expanded += 1
        if expanded > max_expanded:
            status = "aborted"
            break

        for kind, nxt in successors(loc, pp, wh, pdl):
            step = _d(prob, loc, nxt)
            if step == INF:
                continue
            t2 = 0.0
            if use_time:
                t2 = t + (0.0 if loc == nxt else prob.duration.get((loc, nxt), INF))
                if kind == "pickup":
                    opens, closes = pick_win[nxt]
                    t2 = max(t2, opens)
                    if t2 > closes:
                        continue
                elif kind == "deliver":
                    opens, closes = del_win[nxt]
                    t2 = max(t2, opens)
                    if t2 > closes:
                        continue
            if kind == "pickup":
                new_state = (nxt, pp - {nxt}, wh, pdl)
            elif kind == "warehouse":
                new_state = (nxt, pp, True, pdl)
            elif kind == "deliver":
                new_state = (nxt, pp, wh, pdl - {nxt})
            else:
                new_state = (nxt, pp, wh, pdl)
            g2 = g + step
            hv = h(nxt, new_state[1] | new_state[3])
            priority = g2 + hv if astar else hv
            generated += 1
            heapq.heappush(heap, (priority, next(counter), g2, t2, new_state, path + ((kind, nxt),)))

    seconds = time.perf_counter() - t_begin
    if best is None:
        return SearchResult(vehicle_id, algorithm, heuristic, [], INF, expanded, generated, seconds, status)
    route = _build_route(best[1], pick_items, del_items, item_ids, depot)
    return SearchResult(vehicle_id, algorithm, heuristic, route, best[0], expanded, generated, seconds, status)


def solve(prob, algorithm="astar", heuristic="h1", assignment=None, use_time=False,
          start_time=None, max_expanded=500_000):
    """Solve every vehicle. Returns (State, list of SearchResult)."""
    assignment = assignment or split_items(prob)
    routes, results = {}, []
    for vid, ids in assignment.items():
        res = solve_vehicle(prob, vid, ids, algorithm, heuristic, use_time, start_time, max_expanded)
        results.append(res)
        routes[vid] = res.route
    return State(routes=routes), results


def run_all(prob, assignment=None, use_time=False, **kwargs):
    """Greedy and A* with h1 and h2. Returns (comparison DataFrame, {(algorithm, heuristic): State})."""
    rows, states = [], {}
    for algo in ("greedy", "astar"):
        for hname in ("h1", "h2"):
            state, res = solve(prob, algo, hname, assignment, use_time, **kwargs)
            solved = all(r.status == "solved" for r in res)
            feasible = solved and prob.is_feasible(state, check_capacity=True)
            if feasible and use_time and hasattr(prob, "is_feasible_full"):
                feasible = prob.is_feasible_full(state)
            rows.append({
                "algorithm": algo,
                "heuristic": hname,
                "cost_km": round(sum(r.cost for r in res), 3) if solved else float("nan"),
                "expanded_nodes": sum(r.expanded for r in res),
                "generated_nodes": sum(r.generated for r in res),
                "seconds": round(sum(r.seconds for r in res), 4),
                "status": "solved" if solved else ",".join(sorted({r.status for r in res})),
                "feasible": feasible,
            })
            states[(algo, hname)] = state
    return pd.DataFrame(rows), states
