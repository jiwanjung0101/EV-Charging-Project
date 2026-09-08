"""
Entry point. Loads the data, runs the uncoordinated baseline, the three
scheduling schemes and the cost-carbon sweep, prints the reports, and writes
every figure and the performance table.

Edit CONFIG below for model parameters, and NODE_DATE_MAP in
scheduler/data_loader.py to change which carbon date each node uses.
"""

from __future__ import annotations

import math

from scheduler.data_loader import (
    CHARGING_NODES,
    NODE_DATE_MAP,
    get_node_positions,
    load_evs,
    load_nodal_carbon,
    load_nodal_prices,
)
from scheduler.plot import plot_all
from scheduler.results import run_all_schemes

CONFIG = dict(
    n_nodes        = 4,      # CAISO nodes used
    periods        = 24,     # scheduling horizon, hours
    cap_fraction   = 0.85,   # tightest carbon cap, as a fraction of the baseline
    v2g_enabled    = True,
    eta_c          = 0.95,   # charging efficiency
    eta_d          = 0.95,   # discharging efficiency
    deg_cost       = 0.02,   # battery degradation, $/kWh discharged
    grid_cap_kw    = 85.0,   # per-node power cap
    interval_hours = 1.0,
    max_distance   = 4.0,    # furthest an EV may be assigned, grid units
    out_dir        = "plots",
)


def _pos(x, y) -> str:
    return f"({x}, {y})"


def report_inputs(evs, carbon, nodes) -> None:
    print("\nCarbon intensity date per node:")
    for node, date in NODE_DATE_MAP.items():
        if node in nodes:
            values = carbon[node].values()
            print(f"  {node:<28} {date}  "
                  f"({min(values):.0f} to {max(values):.0f} g CO2/kWh)")

    print("\nCharging node positions:")
    for node in CHARGING_NODES:
        if node.name in nodes:
            print(f"  {node.name:<28} {_pos(node.grid_x, node.grid_y)}")

    print("\nEV positions:")
    for ev in evs:
        print(f"  EV {ev.name:<4} {_pos(ev.grid_x, ev.grid_y):<9} "
              f"arrives {ev.arrival:>2}, departs {ev.departure:>2}, "
              f"starts at {ev.arrival_energy:.1f} kWh")


def report_assignment(scheme, evs, node_positions) -> None:
    """Print where each EV charges under one scheme, and how far it travels."""
    print(f"\nNode assignment, {scheme.label}:")
    print(f"  {'EV':<5} {'EV pos':<9} {'Assigned node':<28} {'Node pos':<9} {'Dist':>6}")
    print("  " + "-" * 60)

    for ev in evs:
        node = scheme.assignment[ev.name]
        x, y = node_positions[node]
        distance = math.dist((ev.grid_x, ev.grid_y), (x, y))
        print(f"  {ev.name:<5} {_pos(ev.grid_x, ev.grid_y):<9} {node:<28} "
              f"{_pos(x, y):<9} {distance:>6.2f}")

    counts: dict[str, int] = {}
    for node in scheme.assignment.values():
        counts[node] = counts.get(node, 0) + 1
    print("  EVs per node: " + ", ".join(
        f"{node} {count}" for node, count in sorted(counts.items())))
    print(f"  Fleet average distance: {scheme.avg_distance:.4f} grid units")


def report_comparison(scheme, baseline) -> None:
    """Print one scheme side by side with the uncoordinated baseline."""
    def change(value, base):
        return f"{(value - base) / abs(base) * 100:+.1f}%" if abs(base) > 1e-9 else "n/a"

    metrics = [
        ("Total cost ($)",          scheme.total_cost,    baseline.total_cost),
        ("  energy ($)",            scheme.energy_cost,   baseline.energy_cost),
        ("  degradation ($)",       scheme.degradation,   baseline.degradation),
        ("Emissions (g CO2)",       scheme.emissions,     baseline.emissions),
        ("Net energy (kWh)",        scheme.net_energy,    baseline.net_energy),
        ("Avg intensity (g/kWh)",   scheme.avg_intensity, baseline.avg_intensity),
        ("Avg EV to node distance", scheme.avg_distance,  baseline.avg_distance),
    ]

    print(f"\n{scheme.label} vs uncoordinated baseline")
    print(f"  {'Metric':<24} {'Scheme':>12} {'Baseline':>12} {'Change':>9}")
    print("  " + "-" * 59)
    for name, value, base in metrics:
        print(f"  {name:<24} {value:>12.2f} {base:>12.2f} {change(value, base):>9}")


def main() -> None:
    nodal_df, nodes, time_list = load_nodal_prices(
        n_nodes=CONFIG["n_nodes"], periods=CONFIG["periods"],
    )
    carbon = load_nodal_carbon(periods=CONFIG["periods"])
    evs    = load_evs()
    node_positions = get_node_positions()

    print(f"Loaded {len(evs)} EVs, {len(nodes)} nodes, {len(time_list)} periods.")
    print(f"Distance cap: {CONFIG['max_distance']} grid units")
    report_inputs(evs, carbon, nodes)

    results, sweep = run_all_schemes(
        evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
        cap_fraction=CONFIG["cap_fraction"],
        v2g_enabled=CONFIG["v2g_enabled"],
        eta_c=CONFIG["eta_c"], eta_d=CONFIG["eta_d"],
        deg_cost=CONFIG["deg_cost"], grid_cap_kw=CONFIG["grid_cap_kw"],
        interval_hours=CONFIG["interval_hours"],
        max_distance=CONFIG["max_distance"],
    )
    baseline, _, _, balanced = results

    report_assignment(baseline, evs, node_positions)
    report_assignment(balanced, evs, node_positions)
    report_comparison(balanced, baseline)

    plot_all(
        results=results, sweep=sweep, evs=evs, nodal_df=nodal_df,
        carbon=carbon, time_list=time_list, charging_nodes=CHARGING_NODES,
        grid_cap_kw=CONFIG["grid_cap_kw"], out_dir=CONFIG["out_dir"],
    )
    print(f"All outputs written to '{CONFIG['out_dir']}/'")


if __name__ == "__main__":
    main()
