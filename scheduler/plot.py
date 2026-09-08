"""
Figures and the performance table. Call plot_all() to generate everything.
carbon is a nodal dict {node: {period: g CO2/kWh}}.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# One colour per node, kept consistent across figures.
NODE_COLORS = ["#2CA25F", "#E6550D", "#2171B5", "#C94040", "#7B55A8"]

CHARGE_COLOR = "#2171B5"
PRICE_COLOR  = "#E6550D"
CARBON_COLOR = "#2CA25F"


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  saved {path}")


def _short(node_name: str) -> str:
    """Drop the -APND suffix for display."""
    return node_name.split("-")[0]


def _apply_assignment(evs, assignment: dict) -> None:
    """Point the shared EV list at one scheme's node assignment."""
    for ev in evs:
        if ev.name in assignment:
            ev.node_id = assignment[ev.name]


def _value(table, key) -> float:
    raw = table.get(key, 0.0)
    return (raw.varValue if hasattr(raw, "varValue") else raw) or 0.0


def _net_load(result, evs_here, time_list: list[int]) -> np.ndarray:
    """Aggregate net power (charge minus discharge) per period, in kW."""
    load = np.zeros(len(time_list))
    for i, t in enumerate(time_list):
        for ev in evs_here:
            if ev.arrival <= t <= ev.departure:
                load[i] += (_value(result.charge, (ev.name, t))
                            - _value(result.discharge, (ev.name, t)))
    return load


def _evs_by_node(evs, assignment: dict) -> dict[str, list]:
    _apply_assignment(evs, assignment)
    grouped: dict[str, list] = defaultdict(list)
    for ev in evs:
        grouped[ev.node_id].append(ev)
    return grouped


def fig_nodal_series(
    series:    dict[str, dict[int, float]],
    order:     list[str],
    time_list: list[int],
    ylabel:    str,
    save_path: str,
) -> plt.Figure:
    """One line per node, used for both the LMP and carbon intensity figures."""
    fig, ax = plt.subplots(figsize=(8.5, 4.0))

    for idx, node in enumerate(order):
        ax.plot(
            time_list, [series[node].get(t, 0.0) for t in time_list],
            color=NODE_COLORS[idx % len(NODE_COLORS)],
            lw=2.0, marker="o", markersize=3.5, label=_short(node),
        )

    ax.set_xlabel("Period (hour)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_grid(
    evs,
    charging_nodes,
    save_path:  str,
    assignment: dict | None = None,
) -> plt.Figure:
    """
    The 10x10 grid with EV and charging node positions. With an assignment,
    EVs are coloured by their node and joined to it by a dashed line.
    """
    grid = 10
    node_color = {
        node.name: NODE_COLORS[i % len(NODE_COLORS)]
        for i, node in enumerate(charging_nodes)
    }
    node_by_name = {node.name: node for node in charging_nodes}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0.5, grid + 0.5)
    ax.set_ylim(0.5, grid + 0.5)
    ax.set_xticks(range(1, grid + 1))
    ax.set_yticks(range(1, grid + 1))
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")

    if assignment:
        for ev in evs:
            node = node_by_name.get(assignment[ev.name])
            if node is not None:
                ax.plot(
                    [ev.grid_x, node.grid_x], [ev.grid_y, node.grid_y],
                    ls="--", lw=1.0, color=node_color[node.name],
                    alpha=0.55, zorder=1,
                )

    for ev in evs:
        color = node_color.get(assignment[ev.name], "#AAAAAA") if assignment else "#AAAAAA"
        ax.scatter(ev.grid_x, ev.grid_y, s=140, color=color, alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(ev.name, (ev.grid_x, ev.grid_y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=11, zorder=4)

    for idx, node in enumerate(charging_nodes):
        color = NODE_COLORS[idx % len(NODE_COLORS)]
        ax.scatter(node.grid_x, node.grid_y, s=320, marker="*", color=color,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(_short(node.name), (node.grid_x, node.grid_y),
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=13, fontweight="bold", color=color, zorder=6)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def fig_node_overlay(
    result,
    series:       dict[str, dict[int, float]],
    time_list:    list[int],
    evs,
    ylabel:       str,
    series_label: str,
    series_color: str,
    save_path:    str,
) -> plt.Figure:
    """
    One panel per node: aggregate net load on the left axis against that node's
    own price or carbon intensity on the right. Negative load is V2G discharge.
    """
    node_evs = _evs_by_node(evs, result.assignment)
    nodes    = sorted(node_evs)

    ncols = 2 if len(nodes) > 1 else 1
    nrows = math.ceil(len(nodes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, node in zip(axes, nodes):
        evs_here = node_evs[node]

        ax.plot(time_list, _net_load(result, evs_here, time_list),
                color=CHARGE_COLOR, lw=2.0, marker="o", markersize=3.5,
                label="Net load (kW)", zorder=3)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel("Net Power (kW)", color=CHARGE_COLOR, fontsize=11)
        ax.tick_params(axis="y", labelcolor=CHARGE_COLOR, labelsize=10)
        ax.set_xlim(time_list[0] - 0.5, time_list[-1] + 0.5)
        ax.set_xlabel("Period (hour)", fontsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.grid(True, axis="y", alpha=0.3, zorder=0)

        twin = ax.twinx()
        twin.plot(time_list, [series[node].get(t, 0.0) for t in time_list],
                  color=series_color, lw=2.0, marker="o", markersize=3.5,
                  label=series_label, zorder=3)
        twin.set_ylabel(ylabel, color=series_color, fontsize=11)
        twin.tick_params(axis="y", labelcolor=series_color, labelsize=10)

        plural = "s" if len(evs_here) != 1 else ""
        ax.set_title(f"Node: {_short(node)}  ({len(evs_here)} EV{plural})",
                     loc="left", fontsize=11)

        handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
        labels  = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
        ax.legend(handles, labels, fontsize=8, loc="upper right", framealpha=0.9)

    for unused in axes[len(nodes):]:
        unused.set_visible(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def fig_node_profiles(
    result,
    time_list:   list[int],
    evs,
    save_path:   str,
    grid_cap_kw: float = 85.0,
) -> plt.Figure:
    """One panel per node with per-EV net power, the node total, and the cap."""
    node_evs = _evs_by_node(evs, result.assignment)
    nodes    = sorted(node_evs)

    fig, axes = plt.subplots(len(nodes), 1, figsize=(9.5, 3.5 * len(nodes)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    colormap = plt.get_cmap("tab20")

    for ax, node in zip(axes, nodes):
        evs_here = node_evs[node]
        total    = np.zeros(len(time_list))

        for idx, ev in enumerate(evs_here):
            net = _net_load(result, [ev], time_list)
            total += net
            ax.plot(time_list, net, lw=0.9, alpha=0.55,
                    color=colormap(idx / max(len(evs_here) - 1, 1)),
                    marker="o", markersize=2, label=f"EV {ev.name}")

        ax.plot(time_list, total, lw=2.2, ls="--", color="black",
                marker="o", markersize=3, label="Node total", zorder=5)
        ax.axhline(grid_cap_kw, color="#C94040", lw=1.6, ls="--",
                   label=f"Node cap ({grid_cap_kw:.0f} kW)", zorder=6)
        ax.axhline(0, color="black", lw=0.7)

        plural = "s" if len(evs_here) != 1 else ""
        ax.set_title(f"Node: {_short(node)}  ({len(evs_here)} EV{plural})",
                     loc="left", fontsize=9)
        ax.set_ylabel("Net Power (kW)", fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.legend(ncol=5, fontsize=7, framealpha=0.85)

    axes[-1].set_xlabel("Period (hour)", fontsize=10)
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def fig_frontier(sweep: list, results: list, save_path: str) -> plt.Figure:
    """
    Total operating cost against emissions as alpha runs from 0 to 1. Cost is
    energy plus degradation, the quantity the LP minimises. The three named
    schemes are marked on top of the swept points.
    """
    costs     = [p.total_cost for p in sweep]
    emissions = [p.emissions for p in sweep]
    alphas    = [p.alpha for p in sweep]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    scatter = ax.scatter(costs, emissions, c=alphas, cmap="RdYlGn_r", s=55,
                         zorder=4, edgecolors="white", linewidths=0.5)
    ax.plot(costs, emissions, color="grey", lw=1.0, ls="--", zorder=3, alpha=0.7)
    fig.colorbar(scatter, ax=ax, pad=0.02).set_label(
        "$\\alpha$ (cost weight)", fontsize=9)

    marks = [
        (1, "Cost-only\n($\\alpha=1$)",        "^", "#1F78B4"),
        (2, "Carbon-only\n($\\beta=1$)",       "v", "#33A02C"),
        (3, "Balanced\n($\\alpha=\\beta=0.5$)", "D", "#E31A1C"),
    ]
    for idx, label, marker, color in marks:
        scheme = results[idx]
        ax.scatter(scheme.total_cost, scheme.emissions, marker=marker, s=140,
                   color=color, zorder=6, edgecolors="black", linewidths=0.7,
                   label=label)

    ax.set_xlabel("Total Operating Cost (\\$), energy + degradation", fontsize=11)
    ax.set_ylabel("Total Carbon Emissions (g CO$_2$)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def table_performance(results: list, out_dir: str) -> None:
    """
    Print the four-scheme table and write it as CSV and LaTeX. Percentages are
    against the uncoordinated baseline, results[0].
    """
    baseline = results[0]

    def pct(value: float, base: float, tex: bool = False) -> str:
        if abs(base) < 1e-9:
            return "n/a"
        sign = "\\%" if tex else "%"
        return f"{(value - base) / abs(base) * 100:+.1f}{sign}"

    rows = [
        {
            "Scheme":          scheme.label,
            "Total Cost ($)":  f"{scheme.total_cost:.2f}",
            "Energy ($)":      f"{scheme.energy_cost:.2f}",
            "Degradation ($)": f"{scheme.degradation:.2f}",
            "Emissions (g)":   f"{scheme.emissions:.2f}",
            "Cost change":     pct(scheme.total_cost, baseline.total_cost),
            "Emissions change": pct(scheme.emissions, baseline.emissions),
            "cost_tex":        pct(scheme.total_cost, baseline.total_cost, tex=True),
            "emissions_tex":   pct(scheme.emissions, baseline.emissions, tex=True),
        }
        for scheme in results
    ]

    os.makedirs(out_dir, exist_ok=True)
    columns = ["Scheme", "Total Cost ($)", "Energy ($)", "Degradation ($)",
               "Emissions (g)", "Cost change", "Emissions change"]

    csv_path = os.path.join(out_dir, "performance_table.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {csv_path}")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Performance comparison across scheduling schemes. Total cost"
        r" is energy cost plus battery degradation, the quantity minimised by"
        r" the scheduler.}",
        r"\label{tab:performance}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Scheme & Total Cost (\$) & Energy (\$) & Deg.\ (\$)"
        r" & Emiss.\ (g CO$_2$) & $\Delta$Cost & $\Delta$Emiss. \\",
        r"\midrule",
    ]
    lines += [
        f"{r['Scheme']} & {r['Total Cost ($)']} & {r['Energy ($)']} "
        f"& {r['Degradation ($)']} & {r['Emissions (g)']} "
        f"& {r['cost_tex']} & {r['emissions_tex']} \\\\"
        for r in rows
    ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(out_dir, "performance_table.tex")
    with open(tex_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"  saved {tex_path}")

    header = (f"  {'Scheme':<26} {'Total($)':>9} {'Energy($)':>10} {'Deg($)':>8} "
              f"{'Emiss.(g)':>12} {'Cost':>8} {'Emiss':>8}")
    print("\n  Performance table")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['Scheme']:<26} {r['Total Cost ($)']:>9} {r['Energy ($)']:>10} "
              f"{r['Degradation ($)']:>8} {r['Emissions (g)']:>12} "
              f"{r['Cost change']:>8} {r['Emissions change']:>8}")
    print()


def plot_all(
    results,
    sweep,
    evs,
    nodal_df,
    carbon:         dict[str, dict[int, float]],
    time_list:      list[int],
    charging_nodes,
    grid_cap_kw:    float = 85.0,
    out_dir:        str = "plots",
) -> None:
    """Generate every figure and the performance table into out_dir."""
    _, cost_only, carbon_only, balanced = results
    path = lambda name: os.path.join(out_dir, name)

    prices = {
        node: {int(t): float(nodal_df.loc[t, node]) for t in nodal_df.index}
        for node in nodal_df.columns
    }

    print("\nGenerating figures")

    plot_grid(evs, charging_nodes, save_path=path("grid_positions.pdf"))
    plot_grid(evs, charging_nodes, save_path=path("fig_grid_map.pdf"),
              assignment=balanced.assignment)

    fig_nodal_series(carbon, sorted(carbon), time_list,
                     "Carbon Intensity (g CO$_2$/kWh)",
                     path("fig_nodal_carbon.pdf"))
    fig_nodal_series(prices, list(nodal_df.columns), time_list,
                     "LMP (\\$/kWh)", path("fig_nodal_prices.pdf"))

    fig_node_overlay(cost_only, prices, time_list, evs,
                     ylabel="LMP (\\$/kWh)", series_label="Node LMP (\\$/kWh)",
                     series_color=PRICE_COLOR,
                     save_path=path("fig_cost_price_overlay.pdf"))
    fig_node_overlay(carbon_only, carbon, time_list, evs,
                     ylabel="Carbon Intensity (g CO$_2$/kWh)",
                     series_label="Carbon intensity (g CO$_2$/kWh)",
                     series_color=CARBON_COLOR,
                     save_path=path("fig_carbon_overlay.pdf"))

    fig_node_profiles(balanced, time_list, evs,
                      save_path=path("fig_balanced_node_profiles.pdf"),
                      grid_cap_kw=grid_cap_kw)

    fig_frontier(sweep, results, path("fig_pareto.pdf"))
    plt.close("all")

    table_performance(results, out_dir)
