"""
Agent-Based Model simulation engine for PolicySim.

Provides the Simulation class for running multi-period agent-based
economic simulations with configurable policy interventions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union

from policysim.agents import EconomicAgent, create_agent_population
from policysim.market import (
    clear_labor_market,
    clear_goods_market,
    compute_gdp,
    compute_unemployment_rate,
    compute_gini,
    compute_poverty_rate,
)
from policysim.policy import (
    TaxPolicy,
    SubsidyPolicy,
    UBiPolicy,
    MinimumWagePolicy,
    InterestRatePolicy,
)


class Simulation:
    """
    Multi-period Agent-Based Model simulation engine.

    Runs economic simulations with heterogeneous agents, market clearing,
    and optional policy interventions. Tracks key macroeconomic metrics
    including Gini coefficient, GDP, unemployment, and poverty rate.

    Parameters
    ----------
    n_agents : int
        Number of agents in the simulation.
    n_periods : int
        Number of simulation periods (steps).
    seed : int, optional
        Random seed for reproducibility.
    employment_rate : float
        Initial employment rate of the population.
    wealth_params : dict, optional
        Parameters for agent wealth distribution.
    skill_params : dict, optional
        Parameters for agent skill distribution.

    Attributes
    ----------
    agents : list of EconomicAgent
        The agent population.
    n_periods : int
        Total simulation periods.
    history : dict
        Time-series data of all tracked metrics.
    """

    def __init__(
        self,
        n_agents: int = 1000,
        n_periods: int = 100,
        seed: Optional[int] = None,
        employment_rate: float = 0.92,
        wealth_params: Optional[Dict[str, float]] = None,
        skill_params: Optional[Dict[str, float]] = None,
    ):
        self.n_agents = n_agents
        self.n_periods = n_periods
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Population parameters
        self.wealth_params = wealth_params or {"wealth_mean": 5000.0, "wealth_std": 3000.0}
        self.skill_params = skill_params or {"skill_mean": 1.0, "skill_std": 0.4}

        # Create initial population
        self.agents = create_agent_population(
            n_agents=n_agents,
            employment_rate=employment_rate,
            seed=seed,
            **self.wealth_params,
            **self.skill_params,
        )

        # Simulation state
        self.current_period = 0
        self.equilibrium_wage = 50.0
        self.equilibrium_price = 1.0

        # History tracking
        self.history: Dict[str, List[float]] = {
            "period": [],
            "gdp": [],
            "gini": [],
            "unemployment": [],
            "poverty_rate": [],
            "mean_wealth": [],
            "median_wealth": [],
            "wage": [],
            "price": [],
            "total_consumption": [],
            "total_savings": [],
            "government_revenue": [],
        }

        # Active policies
        self._active_policies: Dict[int, List[Any]] = {}
        self._min_wage_policy: Optional[MinimumWagePolicy] = None

    def run(
        self,
        policies: Optional[List[Tuple[int, Any]]] = None,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run the simulation with optional policy interventions.

        Each period: agents work -> markets clear -> agents consume -> agents save.
        Policies are applied at their specified period.

        Parameters
        ----------
        policies : list of (int, Policy), optional
            List of (period, policy) tuples. Each policy is applied starting
            at the specified period and remains active.
        verbose : bool
            Whether to print progress messages.

        Returns
        -------
        dict
            History dictionary with numpy arrays of all tracked metrics.
            Keys: 'period', 'gdp', 'gini', 'unemployment', 'poverty_rate',
            'mean_wealth', 'median_wealth', 'wage', 'price',
            'total_consumption', 'total_savings', 'government_revenue'.
        """
        self._reset_history()
        self._active_policies = {}
        self._min_wage_policy = None

        # Organize policies by period
        if policies is not None:
            for period, policy in policies:
                if period not in self._active_policies:
                    self._active_policies[period] = []
                self._active_policies[period].append(policy)

        for t in range(self.n_periods):
            self.current_period = t

            # Activate policies scheduled for this period
            if t in self._active_policies:
                for policy in self._active_policies[t]:
                    if isinstance(policy, MinimumWagePolicy):
                        self._min_wage_policy = policy
                    if verbose:
                        print(f"  Period {t}: Activated {policy}")

            # Step 1: Clear labor market (determine equilibrium wage)
            self.equilibrium_wage = clear_labor_market(
                self.agents,
                min_wage=self._min_wage_policy.level if self._min_wage_policy else 0.0,
            )

            # Step 2: Agents work (earn income)
            for agent in self.agents:
                agent.work(self.equilibrium_wage)

            # Step 3: Apply active policies to agents
            self._apply_policies()

            # Step 4: Clear goods market (determine equilibrium price)
            self.equilibrium_price = clear_goods_market(self.agents)

            # Step 5: Agents consume
            for agent in self.agents:
                agent.consume(self.equilibrium_price)

            # Step 6: Agents save
            for agent in self.agents:
                agent.save()

            # Step 7: Record metrics
            self._record_metrics()

            if verbose and t % max(1, self.n_periods // 10) == 0:
                print(
                    f"  Period {t}: GDP={self.history['gdp'][-1]:.0f}, "
                    f"Gini={self.history['gini'][-1]:.3f}, "
                    f"Unemployment={self.history['unemployment'][-1]:.3f}"
                )

        if verbose:
            print(
                f"Simulation complete. {self.n_periods} periods, "
                f"{self.n_agents} agents."
            )

        return {k: np.array(v) for k, v in self.history.items()}

    def _apply_policies(self) -> None:
        """Apply all currently active policies to agents."""
        # Collect all active policies
        all_active = []
        for period in sorted(self._active_policies.keys()):
            if period <= self.current_period:
                all_active.extend(self._active_policies[period])

        for agent in self.agents:
            for policy in all_active:
                if hasattr(policy, "apply"):
                    # For agent-level policies
                    sig = getattr(policy, "apply")
                    # Check if apply takes agent directly (all policies except MinWage)
                    policy.apply(agent)

    def _record_metrics(self) -> None:
        """Record current period metrics to history."""
        gdp = compute_gdp(self.agents)
        gini = compute_gini(self.agents)
        unemployment = compute_unemployment_rate(self.agents)
        poverty = compute_poverty_rate(self.agents)

        wealth = np.array([a.wealth for a in self.agents])
        consumption = sum(a.consumption for a in self.agents)
        savings = sum(a.savings for a in self.agents)
        gov_revenue = sum(
            p.cost() if hasattr(p, "cost") else 0
            for period_policies in self._active_policies.values()
            for p in period_policies
        )

        self.history["period"].append(self.current_period)
        self.history["gdp"].append(gdp)
        self.history["gini"].append(gini)
        self.history["unemployment"].append(unemployment)
        self.history["poverty_rate"].append(poverty)
        self.history["mean_wealth"].append(float(np.mean(wealth)))
        self.history["median_wealth"].append(float(np.median(wealth)))
        self.history["wage"].append(self.equilibrium_wage)
        self.history["price"].append(self.equilibrium_price)
        self.history["total_consumption"].append(consumption)
        self.history["total_savings"].append(savings)
        self.history["government_revenue"].append(gov_revenue)

    def _reset_history(self) -> None:
        """Reset the history tracking dictionary."""
        for key in self.history:
            self.history[key] = []

    def plot_trajectories(
        self,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None,
        title_prefix: str = "",
    ) -> plt.Figure:
        """
        Plot simulation trajectories for key metrics.

        Creates a 2x3 grid of plots: GDP, Gini, Unemployment, Poverty Rate,
        Wealth (mean/median), and Wage/Price.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height).
        save_path : str, optional
            Path to save the figure.
        title_prefix : str
            Prefix for figure title.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        if len(self.history["period"]) == 0:
            raise ValueError("No simulation data to plot. Run simulation first.")

        periods = np.array(self.history["period"])

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f"{title_prefix}PolicySim — Simulation Trajectories", fontsize=14, fontweight="bold")

        # GDP
        ax = axes[0, 0]
        ax.plot(periods, self.history["gdp"], color="#2196F3", linewidth=1.5)
        ax.set_title("GDP")
        ax.set_xlabel("Period")
        ax.set_ylabel("GDP")
        ax.grid(True, alpha=0.3)

        # Gini
        ax = axes[0, 1]
        ax.plot(periods, self.history["gini"], color="#F44336", linewidth=1.5)
        ax.set_title("Gini Coefficient")
        ax.set_xlabel("Period")
        ax.set_ylabel("Gini")
        ax.axhline(y=0.4, color="gray", linestyle="--", alpha=0.5, label="High inequality")
        ax.grid(True, alpha=0.3)

        # Unemployment
        ax = axes[0, 2]
        ax.plot(periods, np.array(self.history["unemployment"]) * 100, color="#FF9800", linewidth=1.5)
        ax.set_title("Unemployment Rate")
        ax.set_xlabel("Period")
        ax.set_ylabel("Unemployment (%)")
        ax.grid(True, alpha=0.3)

        # Poverty Rate
        ax = axes[1, 0]
        ax.plot(periods, np.array(self.history["poverty_rate"]) * 100, color="#9C27B0", linewidth=1.5)
        ax.set_title("Poverty Rate")
        ax.set_xlabel("Period")
        ax.set_ylabel("Poverty Rate (%)")
        ax.grid(True, alpha=0.3)

        # Wealth
        ax = axes[1, 1]
        ax.plot(periods, self.history["mean_wealth"], color="#4CAF50", linewidth=1.5, label="Mean")
        ax.plot(periods, self.history["median_wealth"], color="#8BC34A", linewidth=1.5, linestyle="--", label="Median")
        ax.set_title("Wealth Distribution")
        ax.set_xlabel("Period")
        ax.set_ylabel("Wealth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Wage & Price
        ax = axes[1, 2]
        ax.plot(periods, self.history["wage"], color="#00BCD4", linewidth=1.5, label="Wage")
        ax2 = ax.twinx()
        ax2.plot(periods, self.history["price"], color="#E91E63", linewidth=1.5, linestyle="--", label="Price")
        ax.set_title("Wage & Price Level")
        ax.set_xlabel("Period")
        ax.set_ylabel("Wage", color="#00BCD4")
        ax2.set_ylabel("Price", color="#E91E63")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def compare_scenarios(
        self,
        baseline_policies: Optional[List[Tuple[int, Any]]] = None,
        treatment_policies: Optional[List[Tuple[int, Any]]] = None,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], plt.Figure]:
        """
        Run and compare baseline vs treatment policy scenarios.

        Parameters
        ----------
        baseline_policies : list, optional
            Policies for the baseline scenario.
        treatment_policies : list, optional
            Policies for the treatment scenario.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Save path for the comparison plot.

        Returns
        -------
        tuple
            (baseline_history, treatment_history, figure)
        """
        # Run baseline
        baseline_sim = Simulation(
            n_agents=self.n_agents,
            n_periods=self.n_periods,
            seed=self.seed,
        )
        baseline_hist = baseline_sim.run(policies=baseline_policies, verbose=False)
        baseline_hist = {k: np.array(v) for k, v in baseline_hist.items()}

        # Run treatment
        treatment_sim = Simulation(
            n_agents=self.n_agents,
            n_periods=self.n_periods,
            seed=self.seed,  # Same seed for comparability
        )
        treatment_hist = treatment_sim.run(policies=treatment_policies, verbose=False)
        treatment_hist = {k: np.array(v) for k, v in treatment_hist.items()}

        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("PolicySim — Baseline vs Policy Scenario", fontsize=14, fontweight="bold")

        metrics = [
            ("gdp", "GDP", "#2196F3"),
            ("gini", "Gini Coefficient", "#F44336"),
            ("unemployment", "Unemployment Rate", "#FF9800"),
            ("poverty_rate", "Poverty Rate", "#9C27B0"),
            ("mean_wealth", "Mean Wealth", "#4CAF50"),
            ("wage", "Wage", "#00BCD4"),
        ]

        for idx, (key, title, color) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            periods = baseline_hist["period"]
            values_base = baseline_hist[key]
            values_treat = treatment_hist[key]

            if key in ("unemployment", "poverty_rate"):
                values_base *= 100
                values_treat *= 100
                ylabel = f"{title} (%)"
            else:
                ylabel = title

            ax.plot(periods, values_base, color=color, linewidth=1.5, alpha=0.6, label="Baseline")
            ax.plot(periods, values_treat, color=color, linewidth=2.0, label="Policy")
            ax.set_title(title)
            ax.set_xlabel("Period")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return baseline_hist, treatment_hist, fig

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics of the simulation results.

        Returns
        -------
        dict
            Summary with 'mean', 'std', 'final', 'initial' for each metric.
        """
        summary = {}
        for key, values in self.history.items():
            if key == "period":
                continue
            arr = np.array(values)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "final": float(arr[-1]),
                "initial": float(arr[0]),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return summary

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.agents = create_agent_population(
            n_agents=self.n_agents,
            seed=self.seed,
            **self.wealth_params,
            **self.skill_params,
        )
        self.current_period = 0
        self.equilibrium_wage = 50.0
        self.equilibrium_price = 1.0
        self._active_policies = {}
        self._min_wage_policy = None
        self._reset_history()
