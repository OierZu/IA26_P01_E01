"""
src/informed.py
Informed search (Greedy best-first and A*) for the pickup-and-delivery problem.

Every product follows producer -> warehouse -> delivery location, so each vehicle works in legs:
    collection leg : depot -> producers -> depot (unload)
    delivery leg   : depot (load) -> customer orders -> depot
Each day a vehicle makes one collection leg and then one delivery leg. The search below
finds the visiting order inside one leg.

Search formulation (one leg)
    State  : (current location, stops still to visit)
    Action : go to one of the stops still to visit (when none is left: go back to the depot)
    Cost   : distance in km between the two locations
    Goal   : no stops left and the vehicle is at the depot

Heuristics (both admissible: they never overestimate the remaining distance)
    h1: distance from the current location to the nearest location still required
    h2: sum, over every location still required, of the cheapest edge that can enter it

Time (use_time=True): service and handling times are added, a vehicle that arrives early
waits for the window to open, and a branch that arrives after the window closes is pruned.
"""
import heapq
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

INF = float("inf")


# --------------------------------------------------------------------------- initial split
def _first_fit(prob, rng=None):
    """
    One attempt of the split. For every day each producer (collection) and each order (delivery)
    goes to the first vehicle that has room and for which the leg still has a solution with the
    time windows. rng=None: heaviest first and least loaded vehicle first; otherwise random order.
    Returns (plan, number of items that fitted nowhere).
    """
    vids = [v.vehicle_id for v in prob.vehicles]
    free = {v: 0.0 for v in vids}     # minute at which each vehicle is free
    ready = {}                        # item -> minute at which it is in the warehouse
    misses = [0]

    def spread(weights, kind, day):
        load = {v: 0.0 for v in vids}
        groups = {v: [] for v in vids}
        keys = sorted(weights, key=lambda k: -weights[k])
        if rng:
            rng.shuffle(keys)
        for key in keys:
            w = weights[key]
            order = sorted(vids, key=lambda v: load[v])
            if rng:
                rng.shuffle(order)
            chosen = None
            for v in order:
                if load[v] + w > prob.veh_map[v].capacity_kg:
                    continue
                items = [i for o in groups[v] + [key] for i in prob.order_items[o]] if kind == "deliver" else []
                start = max([free[v]] + [ready.get(i, 0.0) for i in items])
                if solve_leg(prob, v, kind, groups[v] + [key], day, "astar", "h1", True, start).status == "solved":
                    chosen = v
                    break
            if chosen is None:   # no vehicle works: keep the lightest one (the plan will be infeasible)
                chosen = order[0]
                misses[0] += 1
            load[chosen] += w
            groups[chosen].append(key)
        return groups

    def run(groups, kind, day):       # drive the legs to update the clock of each vehicle
        for v, stops in groups.items():
            items = [i for o in stops for i in prob.order_items[o]] if kind == "deliver" else []
            start = max([free[v]] + [ready.get(i, 0.0) for i in items])
            res = solve_leg(prob, v, kind, stops, day, "astar", "h1", True, start)
            if res.sequence:
                free[v] = res.end_min
                if kind == "collect":
                    for p in res.sequence:
                        for i in prob.items_at(day, p):
                            ready[i] = res.end_min

    plan = {}
    days = sorted({prob.order_pickup_day(o) for o in prob.order_items} |
                  {prob.order_delivery_day(o) for o in prob.order_items})
    for day in days:
        prod_w = {}   # items are collected on their pickup day ...
        for r in prob.requests:
            if r.pickup_start.date() == day:
                prod_w[r.pickup_loc] = prod_w.get(r.pickup_loc, 0.0) + r.weight_kg
        # ... and orders are delivered on their delivery day
        todays = [o for o in sorted(prob.order_items) if prob.order_delivery_day(o) == day]
        old_w = {o: prob.order_weight(o) for o in todays if prob.order_pickup_day(o) < day}
        new_w = {o: prob.order_weight(o) for o in todays if prob.order_pickup_day(o) >= day}

        old = spread(old_w, "deliver", day)       # 1. deliver what is already in the warehouse
        run(old, "deliver", day)
        collect = spread(prod_w, "collect", day)  # 2. collect
        run(collect, "collect", day)
        new = spread(new_w, "deliver", day)       # 3. deliver what was just collected
        run(new, "deliver", day)
        plan[day] = {"collect": collect, "deliver": {v: old[v] + new[v] for v in vids}}
    return plan, misses[0]


def split_plan(prob, tries=300, seed=1):
    """
    Starting split of the work between the vehicles.
    First a deterministic first-fit; if some leg has no solution, random first-fit attempts
    (fixed seed) are repeated until one gives a plan where every leg is solvable.
    Returns {day: {"collect": {vehicle_id: [producer_loc]}, "deliver": {vehicle_id: [order_id]}}}
    """
    rng = random.Random(seed)
    best, best_misses = None, None
    for attempt in range(tries):
        plan, misses = _first_fit(prob, None if attempt == 0 else rng)
        if best is None or misses < best_misses:
            best, best_misses = plan, misses
        if misses == 0:
            break
    return best


# --------------------------------------------------------------------------- heuristics
def make_heuristic(name, prob, places):
    """places: every location that can appear in the leg (stops + depot)."""
    depot = prob.depot_loc
    places = set(places) | {depot}
    # cheapest edge entering each place (used by h2)
    min_in = {s: min((prob.km(u, s) for u in places if u != s), default=0.0) for s in places}

    def required(loc, pend):
        # the depot is needed again if the vehicle is away or has stops left
        req = set(pend)
        if pend or loc != depot:
            req.add(depot)
        return req

    if name == "h1":
        def h(loc, pend):
            return min((prob.km(loc, s) for s in required(loc, pend) if s != loc), default=0.0)
    elif name == "h2":
        def h(loc, pend):
            return sum(min_in[s] for s in required(loc, pend))
    else:
        raise ValueError("heuristic must be 'h1' or 'h2'")
    return h


# --------------------------------------------------------------------------- search
@dataclass
class SearchResult:
    vehicle_id: int
    kind: str            # "collect" or "deliver"
    algorithm: str
    heuristic: str
    sequence: List[int]  # visiting order (producer locations or order ids)
    cost: float
    expanded: int
    generated: int
    seconds: float
    status: str          # "solved", "no_solution" or "aborted"
    end_min: float = 0.0  # minute at which the leg is finished (unload done / back at depot)


def solve_leg(prob, vehicle_id, kind, stops, day=None, algorithm="astar", heuristic="h1",
              use_time=True, start_min=0.0, max_expanded=500_000) -> SearchResult:
    """
    One leg of one vehicle.
    kind='collect': stops are producer locations; kind='deliver': stops are order ids.
    algorithm: 'greedy' (order by h) or 'astar' (order by g + h).
    """
    if algorithm not in ("greedy", "astar"):
        raise ValueError("algorithm must be 'greedy' or 'astar'")
    astar = algorithm == "astar"
    depot = prob.depot_loc
    t_begin = time.perf_counter()

    if not stops:
        return SearchResult(vehicle_id, kind, algorithm, heuristic, [], 0.0, 0, 0, 0.0, "solved", start_min)

    # location and time window of every stop
    if kind == "collect":
        place = {p: p for p in stops}
        window = {p: prob.pickup_window(prob.items_at(day, p)) for p in stops}
    else:
        place = {o: prob.order_loc(o) for o in stops}
        window = {o: prob.delivery_window(prob.order_items[o]) for o in stops}

    h = make_heuristic(heuristic, prob, set(place.values()))

    def places_left(left):
        return {place[s] for s in left}

    # a delivery leg starts by loading the vehicle at the depot
    t0 = start_min + (prob.handling_min if kind == "deliver" else 0.0)

    counter = itertools.count()
    start = (depot, frozenset(stops))
    heap = [(h(depot, places_left(start[1])), next(counter), 0.0, t0, start, ())]
    seen = {}  # state -> list of (g, t) already expanded (a node is skipped only if dominated)
    expanded = generated = 0
    status, best = "no_solution", None

    while heap:
        _, _, g, t, state, path = heapq.heappop(heap)
        if any(g2 <= g and t2 <= t for g2, t2 in seen.get(state, ())):
            continue
        seen.setdefault(state, []).append((g, t))
        loc, left = state

        if not left and loc == depot:
            best, status = (g, t, path), "solved"
            break

        expanded += 1
        if expanded > max_expanded:
            status = "aborted"
            break

        # successors: visit one of the stops left, or go back to the depot
        moves = [(s, place[s]) for s in left] if left else [(None, depot)]
        for s, nxt in moves:
            step = prob.km(loc, nxt)
            if step == INF:
                continue

            t2 = t
            if use_time:
                t2 = t + prob.travel_min(loc, nxt)
                if s is None:                      # back at the depot
                    if kind == "collect":
                        t2 += prob.handling_min    # unloading
                else:
                    opens, closes = window[s]
                    t2 = max(t2, opens)            # wait if early
                    if t2 > closes:
                        continue                   # too late: prune
                    t2 += prob.service_min

            new_state = (nxt, left - {s} if s is not None else left)
            g2 = g + step
            hv = h(nxt, places_left(new_state[1]))
            priority = g2 + hv if astar else hv
            generated += 1
            heapq.heappush(heap, (priority, next(counter), g2, t2, new_state,
                                  path + ((s,) if s is not None else ())))

    seconds = time.perf_counter() - t_begin
    if best is None:
        return SearchResult(vehicle_id, kind, algorithm, heuristic, [], INF, expanded, generated,
                            seconds, status, start_min)
    return SearchResult(vehicle_id, kind, algorithm, heuristic, list(best[2]), best[0],
                        expanded, generated, seconds, status, best[1])


def solve(prob, algorithm="astar", heuristic="h1", assignment=None, use_time=True, max_expanded=500_000):
    """
    Solve every leg. assignment has the same format as split_plan().
    Returns (State, list of SearchResult, plan) where plan is the assignment with the visiting orders found.
    """
    assignment = assignment or split_plan(prob)
    free = {v.vehicle_id: 0.0 for v in prob.vehicles}   # minute at which each vehicle is free
    ready = {}                                           # item -> minute at which it is in the warehouse
    results, plan = [], {}

    def deliver(day, orders_by_vehicle):
        found = {}
        for vid, orders in orders_by_vehicle.items():
            # loading cannot start before every item of the leg is in the warehouse
            items = [i for o in orders for i in prob.order_items[o]]
            start = max([free[vid]] + [ready.get(i, 0.0) for i in items]) if use_time else 0.0
            res = solve_leg(prob, vid, "deliver", orders, day, algorithm, heuristic, use_time, start, max_expanded)
            results.append(res)
            found[vid] = res.sequence
            if res.sequence:
                free[vid] = res.end_min
        return found

    for day in sorted(assignment):
        orders = assignment[day]["deliver"]
        old = {v: [o for o in os if prob.order_pickup_day(o) < day] for v, os in orders.items()}
        new = {v: [o for o in os if prob.order_pickup_day(o) >= day] for v, os in orders.items()}

        old_seq = deliver(day, old)                       # 1. orders already in the warehouse
        collect_seq = {}
        for vid, producers in assignment[day]["collect"].items():   # 2. collection
            res = solve_leg(prob, vid, "collect", producers, day, algorithm, heuristic,
                            use_time, free[vid], max_expanded)
            results.append(res)
            collect_seq[vid] = res.sequence
            if res.sequence:
                free[vid] = res.end_min
                for p in res.sequence:
                    for i in prob.items_at(day, p):
                        ready[i] = res.end_min
        new_seq = deliver(day, new)                       # 3. orders collected today

        plan[day] = {"collect": collect_seq,
                     "deliver": {v: old_seq.get(v, []) + new_seq.get(v, []) for v in orders}}

    return prob.plan_to_state(plan), results, plan


def run_all(prob, assignment=None, use_time=True, **kwargs):
    """
    Greedy and A* with h1 and h2.
    Returns (comparison DataFrame, {(algorithm, heuristic): State}, {(algorithm, heuristic): plan}).
    """
    rows, states, plans = [], {}, {}
    for algo in ("greedy", "astar"):
        for hname in ("h1", "h2"):
            state, res, plan = solve(prob, algo, hname, assignment, use_time, **kwargs)
            solved = all(r.status == "solved" for r in res)
            feasible = False
            if solved:
                feasible = prob.is_feasible(state, check_capacity=True)
                if feasible and use_time:
                    feasible = prob.check_time(state)
            rows.append({
                "algorithm": algo,
                "heuristic": hname,
                "cost_km": round(prob.calculate_cost(state), 3) if solved else float("nan"),
                "expanded_nodes": sum(r.expanded for r in res),
                "generated_nodes": sum(r.generated for r in res),
                "seconds": round(sum(r.seconds for r in res), 4),
                "status": "solved" if solved else ",".join(sorted({r.status for r in res})),
                "feasible": feasible,
            })
            states[(algo, hname)] = state
            plans[(algo, hname)] = plan
    return pd.DataFrame(rows), states, plans
