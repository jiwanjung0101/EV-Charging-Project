"""Loads CAISO nodal LMP prices, per-node carbon intensity, and the EV fleet."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


@dataclass
class ChargingNode:
    """A CAISO node and its fixed position on the 10x10 grid."""
    name:   str    # must match a NODE value in caiso.csv
    grid_x: int
    grid_y: int


# Node positions on the grid. Edit here to move nodes.
CHARGING_NODES: list[ChargingNode] = [
    ChargingNode("CLAP_BUNDLD-APND",         grid_x=3, grid_y=3),
    ChargingNode("POD_DUTCH1_7_UNIT 1-APND", grid_x=8, grid_y=3),
    ChargingNode("POD_SLST13_2_SOLAR1-APND", grid_x=3, grid_y=8),
    ChargingNode("ALAMIT_2_PL1X3-APND",      grid_x=8, grid_y=8),
]

# Carbon intensity date used for each node. Keys must match CHARGING_NODES
# names, values must be TRADE_DT entries in carbon_intensity.csv (M/D/YYYY).
NODE_DATE_MAP: dict[str, str] = {
    "CLAP_BUNDLD-APND":         "2/17/2026",
    "POD_DUTCH1_7_UNIT 1-APND": "2/18/2026",
    "POD_SLST13_2_SOLAR1-APND": "2/19/2026",
    "ALAMIT_2_PL1X3-APND":      "2/20/2026",
}


def get_node_positions() -> dict[str, tuple[int, int]]:
    return {n.name: (n.grid_x, n.grid_y) for n in CHARGING_NODES}


@dataclass
class EV:
    """One electric vehicle. node_id is None until Stage 1 assigns a node."""
    name:                  str
    arrival:               int
    departure:             int
    arrival_energy:        float
    desired_energy:        float
    battery_capacity:      float = 40.0
    max_charging_power:    float = 12.0
    max_discharging_power: float = 4.0
    grid_x:                int = 0
    grid_y:                int = 0
    node_id:               str | None = None


def load_nodal_prices(
    n_nodes:  int = 4,
    periods:  int = 24,
    filename: str = "caiso.csv",
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """
    Load CAISO day-ahead LMPs. Returns a DataFrame indexed by hour with one
    column per node in $/kWh, the node names, and the list of periods. Node
    order follows CHARGING_NODES so grid positions stay in sync.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = df[df["LMP_TYPE"] == "LMP"]
    df = df[["OPR_HR", "NODE", "MW"]].rename(
        columns={"OPR_HR": "Hour", "NODE": "Node", "MW": "Price_MWh"}
    )

    # Keep only nodes with a complete 24-hour record.
    hours_per_node = df.groupby("Node")["Hour"].nunique()
    full_nodes = set(hours_per_node[hours_per_node >= periods].index)

    selected = [n.name for n in CHARGING_NODES if n.name in full_nodes][:n_nodes]

    df = df[df["Node"].isin(selected) & (df["Hour"] <= periods)].copy()
    df["Price_kWh"] = df["Price_MWh"] / 1000.0

    prices = (
        df.pivot_table(index="Hour", columns="Node", values="Price_kWh")
        .sort_index()[selected]
    )
    return prices, selected, list(range(1, periods + 1))


def load_nodal_carbon(
    filename:      str = "carbon_intensity.csv",
    periods:       int = 24,
    node_date_map: dict[str, str] | None = None,
) -> dict[str, dict[int, float]]:
    """
    Load one carbon intensity profile per node. Each node takes the 24-hour
    AVG_EM_RATE series for its date in node_date_map, converted from
    MTCO2e/MWh to g CO2/kWh. Returns {node: {period: g CO2/kWh}}.
    """
    node_date_map = node_date_map or NODE_DATE_MAP
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    available = set(df["TRADE_DT"].unique())

    carbon: dict[str, dict[int, float]] = {}
    for node, date_str in node_date_map.items():
        if date_str not in available:
            raise ValueError(
                f"Date '{date_str}' for node '{node}' is not in {filename}. "
                f"Available dates: {sorted(available)}"
            )
        day = (
            df[df["TRADE_DT"] == date_str]
            .sort_values("TRADE_HR")
            .head(periods)
        )
        carbon[node] = {
            t: rate * 1000.0
            for t, rate in enumerate(day["AVG_EM_RATE"], start=1)
        }
    return carbon


def load_evs(filename: str = "ev_infoV5.csv") -> list[EV]:
    """
    Load the EV fleet. Required columns: EV, Arrival Time, Departure Time,
    Arrival Energy, Desired Energy, X, Y. Battery Capacity, Max Charging Power
    and Max Discharging Power are optional and fall back to the EV defaults.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, filename))

    missing = {"X", "Y"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing column(s) {sorted(missing)}. "
            "Add integer grid positions (1-10) for every EV and re-run."
        )

    # to_dict keeps each column's own dtype, which iterrows would flatten to
    # float and turn EV 5 into "5.0". Arrival and departure are truncated to
    # the hour because the horizon is hourly.
    return [
        EV(
            name                  = str(row["EV"]),
            arrival               = int(row["Arrival Time"]),
            departure             = int(row["Departure Time"]),
            arrival_energy        = float(row["Arrival Energy"]),
            desired_energy        = float(row["Desired Energy"]),
            battery_capacity      = float(row.get("Battery Capacity", 40.0)),
            max_charging_power    = float(row.get("Max Charging Power", 12.0)),
            max_discharging_power = float(row.get("Max Discharging Power", 4.0)),
            grid_x                = int(row["X"]),
            grid_y                = int(row["Y"]),
        )
        for row in df.to_dict("records")
    ]
