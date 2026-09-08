"""
Stage 1 charger allocation and the Stage 2 charge/discharge linear program.

carbon is a nodal dict {node: {period: g CO2/kWh}}. Price and carbon are each
divided by their maximum over all nodes and periods, so negative LMPs stay
negative and every period keeps its ratio to the raw signal.
"""

from __future__ import annotations

import math

import pandas as pd
import pulp as lp

PENALTY = 1e6   # objective cost per gram of carbon-cap slack


def active_slots(ev, time_list: list[int]) -> list[int]:
    """Periods in which the EV is plugged in."""
    return [t for t in time_list if ev.arrival <= t <= ev.departure]


def is_active(ev, t: int) -> bool:
    return ev.arrival <= t <= ev.departure


def _max_carbon(carbon: dict[str, dict[int, float]]) -> float:
    return max(v for schedule in carbon.values() for v in schedule.values())


def _effective_carbon_cap(beta: float, cap_fraction: float | None) -> float | None:
    """
    Cap fraction phi(beta), tightening as carbon weight rises:

        beta = 0        no cap
        0 < beta < 1    1 - beta * (1 - phi_0)
        beta >= 1       phi_0
    """
    if cap_fraction is None or beta <= 0.0:
        return None
    if beta >= 1.0:
        return cap_fraction
    return 1.0 - beta * (1.0 - cap_fraction)


def assign_ev_nodes(
    evs,
    nodal_df:       pd.DataFrame,
    carbon:         dict[str, dict[int, float]],
    time_list:      list[int],
    alpha:          float = 0.5,
    beta:           float = 0.5,
    grid_cap_kw:    float = 85.0,
    interval_hours: float = 1.0,
    eta_c:          float = 0.95,
    node_positions: dict[str, tuple[float, float]] | None = None,
    max_distance:   float | None = 4.0,
) -> tuple[dict, dict]:
    """
    Stage 1. Give every EV exactly one charging node.

    Each (EV, node) pair within max_distance is scored by the integrated
    charging signal alpha * price + beta * carbon, summed over the EV's active
    window. Pairs are taken in ascending score order and accepted only if the
    node still has peak-power headroom in every active period and enough energy
    throughput over the window to fully charge the EV. Any EV that passes no
    node falls back to the reachable node with the most headroom.

    Returns (nodal_prices, assignment), where nodal_prices maps (node, period)
    to $/kWh and assignment maps EV name to node name.
    """
    nodes  = list(nodal_df.columns)
    ev_map = {ev.name: ev for ev in evs}

    p_max = float(nodal_df.values.max())
    c_max = _max_carbon(carbon)

    def reachable_nodes(ev) -> list[str]:
        if max_distance is None or node_positions is None:
            return nodes
        within = [
            n for n in nodes
            if math.dist((ev.grid_x, ev.grid_y), node_positions[n]) <= max_distance
        ]
        if within:
            return within
        print(f"  EV {ev.name}: no node within {max_distance} units, "
              f"distance constraint relaxed")
        return nodes

    # committed[(node, t)] is the peak charging power already promised there.
    committed: dict[tuple[str, int], float] = {}
    assignment: dict[str, str] = {}

    def commit(ev, node: str, slots: list[int]) -> None:
        ev.node_id = node
        assignment[ev.name] = node
        for t in slots:
            committed[(node, t)] = committed.get((node, t), 0.0) + ev.max_charging_power

    # Score every reachable pair, then work through them cheapest first.
    candidates: list[tuple[float, str, str]] = []
    for ev in evs:
        for node in reachable_nodes(ev):
            score = sum(
                alpha * nodal_df.loc[t, node] / p_max
                + beta * carbon[node].get(t, 0.0) / c_max
                for t in active_slots(ev, time_list)
                if t in nodal_df.index
            )
            candidates.append((score, ev.name, node))
    candidates.sort(key=lambda pair: pair[0])

    for _, ev_name, node in candidates:
        if ev_name in assignment:
            continue
        ev    = ev_map[ev_name]
        slots = active_slots(ev, time_list)

        peak_ok = all(
            committed.get((node, t), 0.0) + ev.max_charging_power <= grid_cap_kw
            for t in slots
        )
        if not peak_ok:
            continue

        deliverable = sum(
            (grid_cap_kw - committed.get((node, t), 0.0)) * interval_hours
            for t in slots
        )
        if deliverable < (ev.desired_energy - ev.arrival_energy) / eta_c - 1e-6:
            continue

        commit(ev, node, slots)

    # Fallback: place anything left where the most headroom remains.
    for ev in evs:
        if ev.name in assignment:
            continue
        slots = active_slots(ev, time_list)
        best_node = max(
            reachable_nodes(ev),
            key=lambda n: sum(
                max(0.0, grid_cap_kw - committed.get((n, t), 0.0)) * interval_hours
                for t in slots
            ),
        )
        commit(ev, best_node, slots)
        print(f"  EV {ev.name} failed every capacity check, "
              f"fallback to {best_node}")

    nodal_prices = {
        (node, int(t)): float(nodal_df.loc[t, node])
        for node in nodes
        for t in nodal_df.index
    }
    return nodal_prices, assignment


def run_scheduler(
    carbon:            dict[str, dict[int, float]],
    time_list:         list[int],
    evs,
    nodal_prices:      dict[tuple[str, int], float],
    carbon_baseline_g: float,
    interval_hours:    float = 1.0,
    alpha:             float = 0.5,
    beta:              float = 0.5,
    cap_fraction:      float | None = 0.85,
    v2g_enabled:       bool  = True,
    eta_c:             float = 0.95,
    eta_d:             float = 0.95,
    deg_cost:          float = 0.02,
    grid_cap_kw:       float = 85.0,
):
    """
    Stage 2. Minimise the signal-weighted net charging cost plus battery
    degradation, subject to battery dynamics, departure state of charge,
    per-node power caps, and a soft carbon cap.

    Node assignments come from Stage 1 and are held fixed. Only the carbon cap
    is soft: it is relaxed by slack priced at PENALTY. carbon_baseline_g is the
    uncoordinated schedule's emissions, which the cap fraction multiplies.

    Returns (charge, discharge, status, diagnostics).
    """
    model = lp.LpProblem("EV_Scheduler", lp.LpMinimize)

    charge = lp.LpVariable.dicts(
        "charge", ((ev.name, t) for ev in evs for t in time_list), lowBound=0
    )
    discharge = lp.LpVariable.dicts(
        "discharge", ((ev.name, t) for ev in evs for t in time_list), lowBound=0
    )
    slack = lp.LpVariable("carbon_slack", lowBound=0)

    p_max = max(nodal_prices.values())
    c_max = _max_carbon(carbon)

    def signal(ev, t: int) -> float:
        """Integrated charging signal at this EV's node and period."""
        price  = nodal_prices[(ev.node_id, t)] / p_max
        intens = carbon[ev.node_id].get(t, 0.0) / c_max
        return alpha * price + beta * intens

    # Degradation is put on the same normalised scale as the signal so the two
    # objective terms stay comparable.
    deg_norm = deg_cost / p_max

    model += (
        lp.lpSum(
            (
                signal(ev, t) * (charge[(ev.name, t)] - discharge[(ev.name, t)])
                + deg_norm * discharge[(ev.name, t)]
            ) * interval_hours
            for ev in evs
            for t in time_list
        )
        + PENALTY * slack
    )

    # Per-node power cap.
    nodes: dict[str, list] = {}
    for ev in evs:
        nodes.setdefault(ev.node_id, []).append(ev)

    for node, node_evs in nodes.items():
        for t in time_list:
            model += (
                lp.lpSum(charge[(ev.name, t)] - discharge[(ev.name, t)]
                         for ev in node_evs) <= grid_cap_kw,
                f"NodeCap_{node}_{t}",
            )

    # Fleet-wide soft carbon cap, scaled by the carbon weight.
    cap = _effective_carbon_cap(beta, cap_fraction)
    cap_g = None
    if cap is None:
        print(f"  carbon cap off (beta={beta:.2f})")
    else:
        cap_g = cap * carbon_baseline_g
        print(f"  carbon cap: {cap:.3f} x {carbon_baseline_g:,.0f} g "
              f"= {cap_g:,.0f} g CO2")
        model += (
            lp.lpSum(
                (charge[(ev.name, t)] - discharge[(ev.name, t)])
                * carbon[ev.node_id].get(t, 0.0) * interval_hours
                for ev in evs
                for t in time_list
            ) <= cap_g + slack,
            "CarbonCap",
        )

    # Battery dynamics and per-EV limits.
    for ev in evs:
        slots = active_slots(ev, time_list)

        for t in time_list:
            if not is_active(ev, t):
                model += charge[(ev.name, t)] == 0,    f"OffC_{ev.name}_{t}"
                model += discharge[(ev.name, t)] == 0, f"OffD_{ev.name}_{t}"
                continue
            model += charge[(ev.name, t)] <= ev.max_charging_power, f"MaxC_{ev.name}_{t}"
            if v2g_enabled:
                model += (discharge[(ev.name, t)] <= ev.max_discharging_power,
                          f"MaxD_{ev.name}_{t}")
            else:
                model += discharge[(ev.name, t)] == 0, f"NoV2G_{ev.name}_{t}"

        if not slots:
            continue

        energy = lp.LpVariable.dicts(
            f"E_{ev.name}", slots, lowBound=0, upBound=ev.battery_capacity
        )
        model += energy[slots[0]] == ev.arrival_energy, f"InitSOC_{ev.name}"
        for prev_t, t in zip(slots, slots[1:]):
            model += (
                energy[t] == energy[prev_t]
                + (charge[(ev.name, t)] * eta_c
                   - discharge[(ev.name, t)] / eta_d) * interval_hours,
                f"SOC_{ev.name}_{t}",
            )
        model += energy[slots[-1]] >= ev.desired_energy, f"DepSOC_{ev.name}"

    model.solve(lp.PULP_CBC_CMD(msg=0))

    slack_g = slack.varValue or 0.0
    if slack_g > 1e-4:
        print(f"  carbon cap exceeded by {slack_g:,.2f} g CO2 (slack used)")

    diagnostics = {"cap_g": cap_g, "slack_g": slack_g}
    return charge, discharge, lp.LpStatus[model.status], diagnostics


def run_baseline(
    evs,
    time_list:      list[int],
    charging_nodes,
    interval_hours: float = 1.0,
    eta_c:          float = 0.95,
):
    """
    Uncoordinated reference. Each EV takes its closest node and charges at
    maximum power from arrival until it reaches its departure energy. No LP and
    no V2G. Returns (charge, discharge, assignment).
    """
    assignment: dict[str, str] = {}
    charge:    dict[tuple[str, int], float] = {}
    discharge: dict[tuple[str, int], float] = {}

    for ev in evs:
        node = min(
            charging_nodes,
            key=lambda n: math.dist((ev.grid_x, ev.grid_y), (n.grid_x, n.grid_y)),
        )
        ev.node_id = node.name
        assignment[ev.name] = node.name

        soc = ev.arrival_energy
        for t in time_list:
            discharge[(ev.name, t)] = 0.0
            if not is_active(ev, t) or soc >= ev.desired_energy:
                charge[(ev.name, t)] = 0.0
                continue
            needed = (ev.desired_energy - soc) / eta_c
            power  = min(ev.max_charging_power, needed / interval_hours)
            soc   += power * eta_c * interval_hours
            charge[(ev.name, t)] = power

    return charge, discharge, assignment


def compute_metrics(
    carbon:         dict[str, dict[int, float]],
    time_list:      list[int],
    charge,
    discharge,
    evs,
    nodal_prices:   dict[tuple[str, int], float],
    interval_hours: float = 1.0,
    deg_cost:       float = 0.02,
) -> tuple[float, float, float, float, float]:
    """
    Score a schedule at each EV's assigned node. Accepts LP variables or plain
    floats. Returns (energy cost $, emissions g CO2, net energy kWh,
    average intensity g/kWh, degradation cost $).
    """
    def value(table, key):
        raw = table.get(key, 0.0)
        return (raw.varValue if hasattr(raw, "varValue") else raw) or 0.0

    cost = emissions = energy = degradation = 0.0

    for ev in evs:
        for t in active_slots(ev, time_list):
            discharged = value(discharge, (ev.name, t))
            net = (value(charge, (ev.name, t)) - discharged) * interval_hours

            cost        += net * nodal_prices[(ev.node_id, t)]
            emissions   += net * carbon[ev.node_id].get(t, 0.0)
            energy      += net
            degradation += discharged * interval_hours * deg_cost

    intensity = emissions / energy if energy > 1e-9 else 0.0
    return cost, emissions, energy, intensity, degradation
