"""
Runs the uncoordinated baseline, the three evaluation schemes, and the
cost-carbon sweep. Every optimised point re-runs both stages: EVs are rescored
and reallocated under that weighting, then the LP is solved against the result.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import pandas as pd

from scheduler.data_loader import CHARGING_NODES, get_node_positions
from scheduler.model import (
    assign_ev_nodes,
    compute_metrics,
    run_baseline,
    run_scheduler,
)

SWEEP_POINTS = 21   # alpha in {0.00, 0.05, ..., 1.00}


@dataclass
class SchemeResult:
    label:           str
    alpha:           float
    beta:            float
    energy_cost:     float
    emissions:       float
    net_energy:      float
    avg_intensity:   float
    degradation:     float
    avg_distance:    float
    charge:          dict = field(repr=False)
    discharge:       dict = field(repr=False)
    nodal_prices:    dict = field(repr=False)
    assignment:      dict = field(repr=False)
    diagnostics:     dict = field(repr=False, default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Energy cost plus battery degradation, the quantity the LP minimises."""
        return self.energy_cost + self.degradation


@dataclass
class SweepPoint:
    alpha:       float
    beta:        float
    energy_cost: float
    emissions:   float
    degradation: float
    cap_g:       float | None
    slack_g:     float

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.degradation


def _avg_distance(evs, assignment: dict, node_positions: dict) -> float:
    if not evs:
        return 0.0
    return sum(
        math.dist((ev.grid_x, ev.grid_y), node_positions[assignment[ev.name]])
        for ev in evs
    ) / len(evs)


def audit_sweep(points: list[SweepPoint]) -> None:
    """
    Report the sweep and check that it traces a non-dominated frontier: total
    cost must fall and emissions must rise as alpha increases. Stage 1 is a
    greedy heuristic, so this holds by observation rather than construction.
    """
    rows = sorted(points, key=lambda p: p.alpha)

    print("\n" + "=" * 72)
    print("  Sweep audit")
    print("=" * 72)
    print(f"  {'alpha':>6} {'energy($)':>10} {'deg($)':>8} {'total($)':>9} "
          f"{'emis(g)':>11} {'cap(g)':>11} {'slack':>7} {'binding':>8}")
    print("  " + "-" * 70)

    for r in rows:
        cap = f"{r.cap_g:,.0f}" if r.cap_g is not None else "off"
        binding = r.cap_g is not None and abs(r.emissions - r.cap_g) < 1.0
        print(f"  {r.alpha:>6.2f} {r.energy_cost:>10.4f} {r.degradation:>8.4f} "
              f"{r.total_cost:>9.4f} {r.emissions:>11,.0f} {cap:>11} "
              f"{r.slack_g:>7.1f} {str(binding):>8}")

    violations = 0
    for a, b in zip(rows, rows[1:]):
        if b.total_cost > a.total_cost + 1e-6:
            print(f"  total cost rose from alpha {a.alpha:.2f} to {b.alpha:.2f}: "
                  f"{a.total_cost:.4f} to {b.total_cost:.4f}")
            violations += 1
        if b.emissions < a.emissions - 1e-3:
            print(f"  emissions fell from alpha {a.alpha:.2f} to {b.alpha:.2f}: "
                  f"{a.emissions:,.0f} to {b.emissions:,.0f}")
            violations += 1

    binding = sum(1 for r in rows
                  if r.cap_g is not None and abs(r.emissions - r.cap_g) < 1.0)
    print(f"\n  Carbon cap binds at {binding}/{len(rows)} points.")
    print("  Frontier is monotone." if violations == 0
          else f"  {violations} monotonicity violation(s) above.")
    print("=" * 72 + "\n")


def run_scheme(
    label:             str,
    alpha:             float,
    beta:              float,
    evs,
    nodal_df:          pd.DataFrame,
    carbon:            dict[str, dict[int, float]],
    time_list:         list[int],
    node_positions:    dict,
    carbon_baseline_g: float,
    cap_fraction:      float | None = 0.85,
    v2g_enabled:       bool  = True,
    eta_c:             float = 0.95,
    eta_d:             float = 0.95,
    deg_cost:          float = 0.02,
    grid_cap_kw:       float = 85.0,
    interval_hours:    float = 1.0,
    max_distance:      float | None = 4.0,
) -> SchemeResult:
    """Run both stages at one (alpha, beta) and score the resulting schedule."""
    evs = [copy.copy(ev) for ev in evs]   # node_id is set per run

    nodal_prices, assignment = assign_ev_nodes(
        evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
        alpha=alpha, beta=beta, grid_cap_kw=grid_cap_kw,
        interval_hours=interval_hours, eta_c=eta_c,
        node_positions=node_positions, max_distance=max_distance,
    )

    charge, discharge, status, diagnostics = run_scheduler(
        carbon=carbon, time_list=time_list, evs=evs,
        nodal_prices=nodal_prices, carbon_baseline_g=carbon_baseline_g,
        interval_hours=interval_hours, alpha=alpha, beta=beta,
        cap_fraction=cap_fraction, v2g_enabled=v2g_enabled,
        eta_c=eta_c, eta_d=eta_d, deg_cost=deg_cost, grid_cap_kw=grid_cap_kw,
    )
    print(f"  [{label}] solver: {status}")

    cost, emissions, energy, intensity, degradation = compute_metrics(
        carbon=carbon, time_list=time_list, charge=charge, discharge=discharge,
        evs=evs, nodal_prices=nodal_prices,
        interval_hours=interval_hours, deg_cost=deg_cost,
    )

    return SchemeResult(
        label=label, alpha=alpha, beta=beta,
        energy_cost=cost, emissions=emissions, net_energy=energy,
        avg_intensity=intensity, degradation=degradation,
        avg_distance=_avg_distance(evs, assignment, node_positions),
        charge=charge, discharge=discharge,
        nodal_prices=nodal_prices, assignment=assignment,
        diagnostics=diagnostics,
    )


def run_all_schemes(
    evs,
    nodal_df:       pd.DataFrame,
    carbon:         dict[str, dict[int, float]],
    time_list:      list[int],
    cap_fraction:   float | None = 0.85,
    v2g_enabled:    bool  = True,
    eta_c:          float = 0.95,
    eta_d:          float = 0.95,
    deg_cost:       float = 0.02,
    grid_cap_kw:    float = 85.0,
    interval_hours: float = 1.0,
    max_distance:   float | None = 4.0,
) -> tuple[list[SchemeResult], list[SweepPoint]]:
    """
    Run the baseline, the three schemes, and the sweep. The carbon cap is
    anchored to the baseline's measured emissions, so the cap fraction
    multiplies the same number at every operating point.

    Returns (results, sweep), where results holds four SchemeResults in the
    order uncoordinated, cost-only, carbon-only, balanced.
    """
    node_positions = get_node_positions()

    print("\n[Uncoordinated] running baseline")
    base_evs = [copy.copy(ev) for ev in evs]
    charge, discharge, assignment = run_baseline(
        evs=base_evs, time_list=time_list, charging_nodes=CHARGING_NODES,
        interval_hours=interval_hours, eta_c=eta_c,
    )
    nodal_prices = {
        (node, int(t)): float(nodal_df.loc[t, node])
        for node in nodal_df.columns
        for t in nodal_df.index
    }
    cost, emissions, energy, intensity, degradation = compute_metrics(
        carbon=carbon, time_list=time_list, charge=charge, discharge=discharge,
        evs=base_evs, nodal_prices=nodal_prices,
        interval_hours=interval_hours, deg_cost=deg_cost,
    )
    baseline = SchemeResult(
        label="Uncoordinated", alpha=float("nan"), beta=float("nan"),
        energy_cost=cost, emissions=emissions, net_energy=energy,
        avg_intensity=intensity, degradation=degradation,
        avg_distance=_avg_distance(base_evs, assignment, node_positions),
        charge=charge, discharge=discharge,
        nodal_prices=nodal_prices, assignment=assignment,
    )

    print(f"  baseline emissions: {emissions:,.2f} g CO2")
    if cap_fraction is not None:
        print(f"  tightest cap (beta=1, phi_0={cap_fraction}): "
              f"{cap_fraction * emissions:,.2f} g CO2")

    shared = dict(
        evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
        node_positions=node_positions, carbon_baseline_g=emissions,
        cap_fraction=cap_fraction, v2g_enabled=v2g_enabled,
        eta_c=eta_c, eta_d=eta_d, deg_cost=deg_cost,
        grid_cap_kw=grid_cap_kw, interval_hours=interval_hours,
        max_distance=max_distance,
    )

    print("\n[Cost-only] running")
    cost_only = run_scheme("Cost-only", alpha=1.0, beta=0.0, **shared)

    print("\n[Carbon-only] running")
    carbon_only = run_scheme("Carbon-only", alpha=0.0, beta=1.0, **shared)

    print("\n[Balanced] running")
    balanced = run_scheme("Balanced", alpha=0.5, beta=0.5, **shared)

    print(f"\n[Sweep] {SWEEP_POINTS} points, both stages re-run at each")
    sweep: list[SweepPoint] = []
    for i in range(SWEEP_POINTS):
        alpha = round(i / (SWEEP_POINTS - 1), 4)
        beta  = round(1.0 - alpha, 4)
        point = run_scheme(f"Sweep a={alpha:.2f}", alpha=alpha, beta=beta, **shared)
        sweep.append(SweepPoint(
            alpha=alpha, beta=beta,
            energy_cost=point.energy_cost, emissions=point.emissions,
            degradation=point.degradation,
            cap_g=point.diagnostics.get("cap_g"),
            slack_g=point.diagnostics.get("slack_g", 0.0),
        ))
        print(f"  alpha={alpha:.2f}  cost={point.total_cost:.2f}  "
              f"emissions={point.emissions:.2f}")

    print("\nAll schemes complete.\n")
    audit_sweep(sweep)
    return [baseline, cost_only, carbon_only, balanced], sweep
