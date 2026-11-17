"""
src/csp.py
Constraint Satisfaction Problem: which vehicle serves each producer visit and each order.

Variables  : one per producer visit ("collect", day, producer_loc) and one per order ("deliver", order_id)
Domains    : the vehicles (node consistency removes vehicles whose capacity is too small)
Constraints (hard)
    vehicle clock: old deliveries of a vehicle are done before its collection, so they delay it
    capacity     : stops of the same leg and vehicle weigh at most the vehicle capacity   (resource / Atmost)
    time windows : stops of the same leg and vehicle must be reachable one after the other
                   inside their windows: (A before B) or (B before A)                    (disjunctive, binary)
    leg check    : the whole leg must have a route that respects all its windows (A*)    (global)
    final test   : with all variables assigned, the complete plan (clock of each vehicle, dwell time
                   of 24 h) must be feasible; if not, the search backtracks
Soft preferences (only used to order the values, never to reject them)
    - perishable products (fruit / vegetables) prefer the refrigerated vehicle
    - a stop prefers the vehicle that already serves a stop close to it (small detour)

Search: backtracking, day by day (later days depend on earlier ones), with MRV + degree (variable order), LCV (value order), forward checking and AC-3.
"""
import time

from src import informed

PERISHABLE = ("fruit", "vegetable")


class CSP:
    def __init__(self, prob):
        self.prob = prob
        depot = prob.depot_loc
        self.vids = [v.vehicle_id for v in prob.vehicles]
        self.vars, self.info, self.group, self.day_of = [], {}, {}, {}

        # ---- collection variables: one per (day, producer)
        visits = {}
        for r in prob.requests:
            visits.setdefault((r.pickup_start.date(), r.pickup_loc), []).append(r.item_id)
        open_of = {}
        for (day, p), items in visits.items():
            var = ("collect", day, p)
            opens, closes = prob.pickup_window(items)
            open_of[(day, p)] = opens
            self._add(var, ("collect", day), p, opens, closes, items, ready=0.0)

        # ---- delivery variables: one per order
        for o in sorted(prob.order_items):
            items = prob.order_items[o]
            day = prob.order_delivery_day(o)
            kind = "old" if prob.order_pickup_day(o) < day else "new"
            opens, closes = prob.delivery_window(items)
            # earliest minute at which all items of the order can be in the warehouse (optimistic)
            ready = 0.0
            for p in {prob.req_map[i].pickup_loc for i in items}:
                pday = prob.order_pickup_day(o)
                first = max(open_of[(pday, p)], prob.travel_min(depot, p))
                ready = max(ready, first + prob.service_min + prob.travel_min(p, depot) + prob.handling_min)
            self._add(("deliver", o), ("deliver", day, kind), prob.order_loc(o), opens, closes, items, ready)

        # members of every group and neighbours of every variable
        self.members = {}
        for var in self.vars:
            self.members.setdefault(self.group[var], []).append(var)
        self.neighbours = {v: [u for u in self.members[self.group[v]] if u != v] for v in self.vars}

        self.domains = {v: [x for x in self.vids if prob.veh_map[x].capacity_kg >= self.info[v]["weight"]]
                        for v in self.vars}      # node consistency (unary capacity constraint)
        self._pair_cache, self._leg_cache, self._partial_cache, self._clock_cache = {}, {}, {}, {}

    def _add(self, var, group, loc, opens, closes, items, ready):
        reqs = [self.prob.req_map[i] for i in items]
        total = sum(r.weight_kg for r in reqs)
        perish = sum(r.weight_kg for r in reqs if r.category in PERISHABLE)
        self.vars.append(var)
        self.group[var] = group
        self.day_of[var] = var[1] if var[0] == "collect" else self.prob.order_delivery_day(var[1])
        self.info[var] = {"loc": loc, "open": opens, "close": closes, "weight": total,
                          "perishable": perish > 0.5 * total, "ready": ready}

    # ------------------------------------------------------------------ constraints
    def _earliest(self, var, load_ready):
        """Earliest minute at which the service of a stop can start (optimistic)."""
        d, prob = self.info[var], self.prob
        if self.group[var][0] == "collect":
            return max(d["open"], prob.travel_min(prob.depot_loc, d["loc"]))
        return max(d["open"], load_ready + prob.handling_min + prob.travel_min(prob.depot_loc, d["loc"]))

    def compatible(self, a, b):
        """Disjunctive constraint: A before B, or B before A, respecting both windows."""
        key = (a, b) if a < b else (b, a)
        if key not in self._pair_cache:
            da, db, prob = self.info[a], self.info[b], self.prob
            ready = max(da["ready"], db["ready"])            # both are loaded together
            ea, eb = self._earliest(a, ready), self._earliest(b, ready)
            gap = prob.service_min + prob.travel_min(da["loc"], db["loc"])
            self._pair_cache[key] = (ea + gap <= db["close"]) or (eb + gap <= da["close"])
        return self._pair_cache[key]

    def leg_ok(self, group, stops):
        """Global constraint: a leg with these stops has a route inside all the windows."""
        key = (group, tuple(sorted(stops)))
        if key not in self._leg_cache:
            prob = self.prob
            if group[0] == "collect":
                res = informed.solve_leg(prob, self.vids[0], "collect", [s[2] for s in stops], group[1],
                                         "astar", "h1", True, 0.0)
            else:
                start = max(self.info[s]["ready"] for s in stops)
                res = informed.solve_leg(prob, self.vids[0], "deliver", [s[1] for s in stops], group[1],
                                         "astar", "h1", True, start)
            self._leg_cache[key] = res.status == "solved"
        return self._leg_cache[key]

    def clock_ok(self, day, x, assignment):
        """
        Vehicle clock constraint: a vehicle first delivers the orders already in the warehouse
        and only then goes to collect, so the old deliveries delay the collection leg.
        """
        olds = sorted(v for v in self.vars if v[0] == "deliver" and self.day_of[v] == day
                      and self.group[v][2] == "old" and assignment.get(v) == x)
        cols = sorted(v for v in self.vars if v[0] == "collect" and v[1] == day and assignment.get(v) == x)
        if not olds or not cols:
            return True
        key = (day, x, tuple(olds), tuple(cols))
        if key not in self._clock_cache:
            start = max(self.info[v]["ready"] for v in olds)
            first = informed.solve_leg(self.prob, x, "deliver", [v[1] for v in olds], day, "astar", "h1", True, start)
            ok = first.status == "solved"
            if ok:
                second = informed.solve_leg(self.prob, x, "collect", [v[2] for v in cols], day,
                                            "astar", "h1", True, first.end_min)
                ok = second.status == "solved"
            self._clock_cache[key] = ok
        return self._clock_cache[key]

    def consistent(self, var, x, assignment):
        """Can var take vehicle x given the stops already assigned to x in the same leg?"""
        mates = [u for u in self.neighbours[var] if assignment.get(u) == x]
        load = self.info[var]["weight"] + sum(self.info[u]["weight"] for u in mates)
        if load > self.prob.veh_map[x].capacity_kg:
            return False
        if any(not self.compatible(var, u) for u in mates):
            return False
        if not self.leg_ok(self.group[var], mates + [var]):
            return False
        return self.clock_ok(self.day_of[var], x, {**assignment, var: x})

    def allowed(self, a, x, b, y):
        """Binary constraint between two variables of the same group (used by AC-3)."""
        if x != y:
            return True
        weights = self.info[a]["weight"] + self.info[b]["weight"]
        return weights <= self.prob.veh_map[x].capacity_kg and self.compatible(a, b)

    # ------------------------------------------------------------------ AC-3
    def ac3(self, domains):
        """Arc consistency (AC-3). Returns False if some domain becomes empty."""
        queue = [(a, b) for a in self.vars for b in self.neighbours[a]]
        while queue:
            a, b = queue.pop(0)
            revised = False
            for x in list(domains[a]):
                if not any(self.allowed(a, x, b, y) for y in domains[b]):   # no support for x
                    domains[a].remove(x)
                    revised = True
            if revised:
                if not domains[a]:
                    return False
                queue += [(c, a) for c in self.neighbours[a] if c != b]
        return True

    # ------------------------------------------------------------------ soft preferences
    def soft_cost(self, var, x, assignment):
        """Lower is better. Only used to break ties between values."""
        cost = 0.0
        if self.info[var]["perishable"] and not self.prob.veh_map[x].refrigerated:
            cost += 5.0                                    # perishable goods prefer the refrigerated vehicle
        mates = [u for u in self.neighbours[var] if assignment.get(u) == x]
        if mates:                                          # small detour: be close to a stop already served
            cost += min(self.prob.km(self.info[var]["loc"], self.info[u]["loc"]) for u in mates)
        else:
            cost += self.prob.km(self.prob.depot_loc, self.info[var]["loc"])
        return cost

    # ------------------------------------------------------------------ plan / goal test
    def to_assignment(self, assignment):
        """Assignment {var: vehicle} -> plan format used by informed.solve (visit orders not decided yet)."""
        plan = {}
        for var, x in assignment.items():
            day = var[1] if var[0] == "collect" else self.prob.order_delivery_day(var[1])
            leg = plan.setdefault(day, {"collect": {v: [] for v in self.vids}, "deliver": {v: [] for v in self.vids}})
            if var[0] == "collect":
                leg["collect"][x].append(var[2])
            else:
                leg["deliver"][x].append(var[1])
        return plan

    def partial_ok(self, assignment):
        """
        Test for the days whose variables are all assigned: every leg of those days
        must be solvable by A* (vehicle clocks and warehouse readiness included).
        Days are checked in order, so day 31 is only tested when day 30 is complete.
        """
        done = []
        for day in sorted(set(self.day_of.values())):
            if all(v in assignment for v in self.vars if self.day_of[v] == day):
                done.append(day)
            else:
                break
        if not done:
            return True
        part = {v: x for v, x in assignment.items() if self.day_of[v] in done}
        key = tuple(sorted((str(v), x) for v, x in part.items()))
        if key not in self._partial_cache:
            _, results, _ = informed.solve(self.prob, "astar", "h1", self.to_assignment(part), True)
            self._partial_cache[key] = all(r.status == "solved" for r in results)
        return self._partial_cache[key]

    def goal_test(self, assignment):
        """Complete plan: every leg solved by A* and the whole schedule feasible."""
        state, results, plan = informed.solve(self.prob, "astar", "h1", self.to_assignment(assignment), True)
        if not all(r.status == "solved" for r in results):
            return None
        return (state, plan) if self.prob.is_feasible_full(state) else None


# ---------------------------------------------------------------------- backtracking search
def backtracking_search(csp, use_mrv=True, use_lcv=True, use_fc=True, use_ac3=True, max_backtracks=20000):
    """
    Backtracking search for the CSP (same structure as BACKTRACK in the lecture notes).
    Returns (assignment or None, stats). stats counts backtracks (assignments that had to be undone).
    """
    stats = {"backtracks": 0, "assignments": 0, "goal_test_failures": 0, "fc_pruned_values": 0, "partial_failures": 0, "stopped_by_limit": False, "seconds": 0.0}
    t0 = time.perf_counter()
    domains = {v: list(d) for v, d in csp.domains.items()}
    stats["initial_domain_size"] = sum(len(d) for d in domains.values())
    if use_ac3 and not csp.ac3(domains):
        stats["seconds"] = round(time.perf_counter() - t0, 3)
        return None, stats
    stats["domain_size_after_ac3"] = sum(len(d) for d in domains.values())
    result = {}

    def select_variable(domains, assignment):
        free = [v for v in csp.vars if v not in assignment]
        first_day = min(csp.day_of[v] for v in free)     # later days depend on earlier days: finish day by day
        free = [v for v in free if csp.day_of[v] == first_day]
        if not use_mrv:
            return free[0]                                      # static order
        # MRV: fewest legal values; degree heuristic as tie-break (most unassigned neighbours)
        return min(free, key=lambda v: (len(domains[v]),
                                        -sum(1 for u in csp.neighbours[v] if u not in assignment)))

    def order_values(var, domains, assignment):
        def lcv(x):   # how many values of the neighbours does x rule out?
            return sum(1 for u in csp.neighbours[var] if u not in assignment
                       for y in domains[u] if not csp.allowed(var, x, u, y))
        key = (lambda x: (lcv(x), csp.soft_cost(var, x, assignment))) if use_lcv \
            else (lambda x: csp.soft_cost(var, x, assignment))
        return sorted(domains[var], key=key)

    def backtrack(assignment, domains):
        if len(assignment) == len(csp.vars):
            found = csp.goal_test(assignment)
            if found is None:
                stats["goal_test_failures"] += 1
                return False
            result["solution"] = found
            return True

        if stats["backtracks"] >= max_backtracks:
            stats["stopped_by_limit"] = True
            return False
        var = select_variable(domains, assignment)
        for x in order_values(var, domains, assignment):
            if not csp.consistent(var, x, assignment):
                continue
            assignment[var] = x
            stats["assignments"] += 1

            new_domains, ok = domains, True
            if use_fc:      # forward checking: prune the neighbours of var
                new_domains = {v: list(d) for v, d in domains.items()}
                new_domains[var] = [x]
                for u in csp.neighbours[var]:
                    if u in assignment:
                        continue
                    before = len(new_domains[u])
                    new_domains[u] = [y for y in new_domains[u] if y != x or csp.consistent(u, y, assignment)]
                    stats["fc_pruned_values"] += before - len(new_domains[u])
                    if not new_domains[u]:
                        ok = False
                        break
            if ok and not csp.partial_ok(assignment):     # a completed day cannot be scheduled
                stats["partial_failures"] += 1
                ok = False
            if ok and backtrack(assignment, new_domains):
                return True
            del assignment[var]                      # undo and try the next value
            stats["backtracks"] += 1
            if stats.get("stopped_by_limit"):
                return False
        return False

    assignment = {}
    solved = backtrack(assignment, domains)
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    if not solved:
        return None, stats
    state, plan = result["solution"]
    stats["assignment"] = dict(assignment)
    return {"assignment": dict(assignment), "state": state, "plan": plan}, stats
