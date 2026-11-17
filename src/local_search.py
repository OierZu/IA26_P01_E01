"""
src/local_search.py
Hill-Climbing for the routing problem.

A solution is a plan (see Problem.plan_to_state):
    plan[day]["collect"][vehicle_id] = [producer_loc, ...]   visit order of the collection leg
    plan[day]["deliver"][vehicle_id] = [order_id, ...]       visit order of the delivery legs
Cost = total distance (km). A neighbour is only accepted if it is feasible
(flow, capacity, time windows and warehouse dwell), so every solution visited is valid.

Neighbourhood (applied to every list of every day):
    relocate : move one stop to another position of the same vehicle
    swap     : exchange two stops of the same vehicle
    2-opt    : reverse a segment of consecutive stops of the same vehicle
    move     : move one stop to another vehicle (any position of its list)
"""
import copy
import time


def _neighbours(plan):
    """Yield (name, new_plan) for every neighbour of a plan."""
    for day in plan:
        for kind in ("collect", "deliver"):
            lists = plan[day][kind]
            for vid, stops in lists.items():
                n = len(stops)

                # relocate: stop i goes to position j
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            new = copy.deepcopy(plan)
                            lst = new[day][kind][vid]
                            lst.insert(j, lst.pop(i))
                            yield "relocate", new

                # swap: stops i and j exchange places
                for i in range(n):
                    for j in range(i + 1, n):
                        new = copy.deepcopy(plan)
                        lst = new[day][kind][vid]
                        lst[i], lst[j] = lst[j], lst[i]
                        yield "swap", new

                # 2-opt: reverse the segment i..j
                for i in range(n):
                    for j in range(i + 2, n):
                        new = copy.deepcopy(plan)
                        lst = new[day][kind][vid]
                        lst[i:j + 1] = reversed(lst[i:j + 1])
                        yield "2-opt", new

                # move: stop i goes to another vehicle
                for i in range(n):
                    for other, other_list in lists.items():
                        if other == vid:
                            continue
                        for j in range(len(other_list) + 1):
                            new = copy.deepcopy(plan)
                            stop = new[day][kind][vid].pop(i)
                            new[day][kind][other].insert(j, stop)
                            yield "move", new


def evaluate(prob, plan):
    """(cost in km, feasible) of a plan."""
    state = prob.plan_to_state(plan)
    if not prob.is_feasible_full(state):
        return float("inf"), False
    return prob.calculate_cost(state), True


def hill_climbing(prob, plan, max_iter=200, max_sideways=0):
    """
    Steepest-ascent hill climbing (here: steepest descent of the cost).
    Each iteration looks at every neighbour and moves to the best feasible one.
    It stops at a local optimum (no better neighbour). With max_sideways > 0 it may also
    cross a plateau: up to that many moves to a neighbour with the same cost (never to a plan
    already visited).
    Returns (best plan, history) where history is a list of dicts (iteration, cost, move).
    """
    cost, ok = evaluate(prob, plan)
    if not ok:
        raise ValueError("the starting plan is not feasible")
    visited = {repr(plan)}
    history = [{"iteration": 0, "cost_km": cost, "move": "start", "neighbours": 0}]
    sideways = 0
    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        best_move, best_plan, best_cost, checked = None, None, float("inf"), 0
        for name, new in _neighbours(plan):
            if repr(new) in visited:
                continue
            checked += 1
            c, ok = evaluate(prob, new)
            if ok and c < best_cost - 1e-9:
                best_move, best_plan, best_cost = name, new, c

        if best_plan is not None and best_cost < cost - 1e-9:
            sideways = 0
        elif best_plan is not None and best_cost <= cost + 1e-9 and sideways < max_sideways:
            sideways += 1                       # plateau: accept an equal-cost neighbour
        else:
            break                               # local optimum
        plan, cost = best_plan, best_cost
        visited.add(repr(plan))
        history.append({"iteration": it, "cost_km": cost, "move": best_move, "neighbours": checked})

    history[0]["seconds"] = round(time.perf_counter() - t0, 3)
    return plan, history
