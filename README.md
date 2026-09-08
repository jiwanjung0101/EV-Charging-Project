# A Locational Price and Carbon Intensity-Aware Charger
Allocation and Charge-Discharge Scheduling Strategy for Electric Vehicle
Fleets* (Jung, Dash, and Srinivasan, 2026).

The scheduler trades electricity cost against carbon emissions using CAISO
nodal prices and per-node grid carbon intensity. The two signals are normalised
by their maxima and combined into one Integrated Charging Signal (ICS),
weighted by alpha for cost and beta for carbon, with alpha + beta = 1.

The pipeline runs in two stages:

1. **Charger allocation.** A capacity-aware greedy procedure assigns each EV to
   a charging node, subject to a distance cap, per-node peak power, and
   energy-throughput feasibility over the EV's parking window.
2. **LP scheduling.** With those assignments fixed, a linear program sets the
   charge and vehicle-to-grid discharge profile of every EV over a 24-hour
   horizon, with a linear battery degradation cost and a soft carbon cap that
   tightens as beta rises.

## Results

Three schemes are evaluated against an uncoordinated baseline that charges each
EV at full power on arrival at its nearest node. Total cost is energy cost plus
battery degradation, which is what the LP minimises.

| Strategy | Total ($) | Energy ($) | Deg. ($) | Savings (%) | Emissions (g CO2) | Reduction (%) |
|---|---:|---:|---:|---:|---:|---:|
| Uncoordinated | 20.57 | 20.57 | 0.00 | - | 130,698.68 | - |
| Cost-only (alpha=1) | 13.67 | 11.83 | 1.84 | 33.6 | 115,465.27 | 11.7 |
| Carbon-only (beta=1) | 19.26 | 16.86 | 2.40 | 6.4 | 98,670.32 | 24.5 |
| Balanced (alpha=beta=0.5) | 14.74 | 12.42 | 2.32 | 28.4 | 102,494.53 | 21.6 |

The balanced scheme cuts total operating cost by 28.4% and carbon emissions by
21.6% at the same time. Sweeping alpha from 0 to 1 in steps of 0.05 traces the
full trade-off frontier, with the knee near the balanced point.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib pulp
```

PuLP ships the CBC solver, so no separate solver install is needed.

## Usage

```bash
python3 main.py
```

The run prints the fleet and node inputs, solves the baseline, the three
schemes and the 21-point sweep, reports the balanced assignment against the
baseline, and writes every figure and the performance table to `plots/`. It
takes under a minute on a laptop.

## Configuration

Edit `CONFIG` at the top of `main.py`:

| Parameter | Meaning | Paper value |
|---|---|---|
| `n_nodes` | CAISO nodes used | 4 |
| `periods` | Scheduling horizon, hours | 24 |
| `cap_fraction` | Tightest carbon cap, as a fraction of the baseline | 0.85 |
| `v2g_enabled` | Allow vehicle-to-grid discharging | True |
| `eta_c` / `eta_d` | Charging / discharging efficiency | 0.95 |
| `deg_cost` | Battery degradation, $/kWh discharged | 0.02 |
| `grid_cap_kw` | Per-node power cap, kW | 85.0 |
| `interval_hours` | Length of one period | 1.0 |
| `max_distance` | Furthest an EV may be assigned, grid units | 4.0 |
| `out_dir` | Where figures and tables are written | `plots` |

Per-EV limits come from `data/ev_infoV5.csv` and default to a 40 kWh battery,
12 kW charging and 4 kW discharging. To change which carbon date a node uses,
edit `NODE_DATE_MAP` in `scheduler/data_loader.py`.

## Data

- **LMP prices** (`data/caiso.csv`): CAISO day-ahead market via OASIS, trade
  date 18 February 2026. Four nodes are used, chosen for price diversity:
  `CLAP_BUNDLD`, `POD_DUTCH1_7_UNIT 1`, `POD_SLST13_2_SOLAR1`, `ALAMIT_2_PL1X3`.
  They sit at symmetric positions on a synthetic 10x10 grid rather than at
  their true geographic locations.
- **Carbon intensity** (`data/carbon_intensity.csv`): CAISO average emissions
  rate report, February 2026. CAISO publishes one California-wide average, so
  each node is given a distinct daily profile: 17 to 20 February map to the
  four nodes in the order above.
- **EV fleet** (`data/ev_infoV5.csv`): 30 synthetic vehicles differing in
  initial state of charge, arrival and departure time, and grid position.

## Layout

```
main.py                      entry point, config and reporting
scheduler/data_loader.py     prices, carbon intensity and fleet
scheduler/model.py           allocation, LP scheduler, baseline, metrics
scheduler/results.py         the four schemes and the alpha sweep
scheduler/plot.py            figures and the performance table
data/                        input CSVs
plots/                       generated figures and tables
```

## Outputs

| File | Contents |
|---|---|
| `fig_nodal_prices.pdf` | Nodal LMP over the day |
| `fig_nodal_carbon.pdf` | Nodal carbon intensity over the day |
| `grid_positions.pdf` | EV and charging node positions |
| `fig_cost_price_overlay.pdf` | Per-node net load against LMP, cost-only scheme |
| `fig_carbon_overlay.pdf` | Per-node net load against carbon intensity, carbon-only scheme |
| `fig_grid_map.pdf` | EV to node assignment, balanced scheme |
| `fig_balanced_node_profiles.pdf` | Per-EV and per-node power, balanced scheme |
| `fig_pareto.pdf` | Cost against emissions frontier over alpha |
| `performance_table.csv` / `.tex` | The results table above |
