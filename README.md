# PolicySim

**Policy Simulation & Counterfactual Analysis Engine**

PolicySim is a professional open-source Python toolkit for designing, simulating, and analyzing economic policies. It combines three powerful methodological pillars:

- **Agent-Based Modeling (ABM)** — Simulate heterogeneous economic agents with realistic behavioral rules
- **Microsimulation** — Track individual-level outcomes under different policy regimes
- **Difference-in-Differences / Synthetic Control** — Estimate causal policy effects using counterfactual methods

---

## Features

- **Heterogeneous Economic Agents** with parametrized wealth, income, consumption propensity, labor supply, and skill levels
- **Labor and Goods Market Clearing** via supply-demand equilibrium with price discovery
- **Rich Policy Toolkit** — progressive/flat tax, targeted subsidies, UBI, minimum wage, monetary policy
- **Full ABM Simulation Engine** — multi-period simulation with configurable agent counts and policy interventions
- **Counterfactual Analysis** — DiD estimator, Synthetic Control Method, placebo tests, event study plots
- **Built-in Metrics** — Gini coefficient, GDP, unemployment rate, inequality indices, poverty rate
- **Visualization** — trajectory plots comparing baseline vs policy scenarios

---

## Installation

```bash
pip install policysim
```

Or from source:

```bash
git clone https://github.com/policysim/policysim.git
cd policysim
pip install -e .
```

---

## Quick Start

```python
from policysim import Simulation, TaxPolicy, UBiPolicy
from policysim.counterfactual import did_estimate

# Run a baseline simulation with 2000 agents over 50 periods
sim = Simulation(n_agents=2000, n_periods=50, seed=42)
df_base = sim.run(policies=None)
sim.plot_trajectories()

# Apply tax reform at period 25
tax_reform = TaxPolicy(rate=0.35, brackets=[(0, 0.10), (50000, 0.25), (150000, 0.40)])
df_policy = sim.run(policies=[(25, tax_reform)])

# Compare outcomes
print(f"Baseline Gini: {df_base['gini'].iloc[-1]:.3f}")
print(f"Policy  Gini: {df_policy['gini'].iloc[-1]:.3f}")
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `policysim.agents` | `EconomicAgent` class with work, consume, save, and policy-response behavior |
| `policysim.market` | Labor market and goods market clearing with equilibrium wage/price discovery |
| `policysim.policy` | Policy intervention classes: tax, subsidy, UBI, minimum wage, interest rate |
| `policysim.abm` | `Simulation` engine — multi-period ABM with metrics tracking and visualization |
| `policysim.counterfactual` | DiD estimator, Synthetic Control, placebo tests, event study analysis |

## Examples

Run the full demonstration:

```bash
python examples/demo.py
```

This runs a complete policy analysis pipeline:
1. ABM simulation with 2000 agents for 50 periods
2. Tax reform at period 25
3. Comparison of Gini, GDP, and unemployment trajectories
4. DiD counterfactual estimation
5. Policy effectiveness summary

---

## Requirements

- Python >= 3.8
- numpy >= 1.22.0
- scipy >= 1.8.0
- matplotlib >= 3.5.0
- pandas >= 1.4.0

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{policysim2024,
  title = {PolicySim: Policy Simulation & Counterfactual Analysis Engine},
  year = {2024},
  url = {https://github.com/policysim/policysim}
}
```
