"""
Economic agents for PolicySim.

Provides the EconomicAgent class representing heterogeneous economic actors
with wealth, income, consumption, labor supply, and policy-response behavior.
"""

import numpy as np
from typing import Dict, Any, Optional


class EconomicAgent:
    """
    Heterogeneous economic agent with behavioral rules.

    Each agent has individual attributes drawn from distributions that
    determine its economic behavior. Agents can work, consume, save, and
    respond to policy interventions.

    Attributes
    ----------
    wealth : float
        Current accumulated wealth.
    income : float
        Current period income (wage * labor supply).
    consumption_propensity : float
        Marginal propensity to consume (0, 1).
    labor_supply : float
        Labor supply in efficiency units.
    skill_level : float
        Productivity/skill multiplier.
    employed : bool
        Whether the agent is currently employed.
    wage : float
        Current wage rate per efficiency unit.
    savings_rate : float
        Fraction of disposable income saved (derived from consumption_propensity).
    """

    def __init__(
        self,
        wealth: float = 1000.0,
        income: float = 0.0,
        consumption_propensity: float = 0.8,
        labor_supply: float = 1.0,
        skill_level: float = 1.0,
        employed: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        wealth : float
            Initial wealth.
        income : float
            Initial income (set to 0 by default; computed during work()).
        consumption_propensity : float
            Fraction of disposable income spent on consumption each period.
        labor_supply : float
            Labor supplied in efficiency units (1.0 = full-time equivalent).
        skill_level : float
            Productivity multiplier relative to baseline (1.0).
        employed : bool
            Initial employment status.
        rng : np.random.Generator, optional
            Random number generator for stochastic behavior.
        """
        self.wealth = float(wealth)
        self.income = float(income)
        self.consumption_propensity = float(np.clip(consumption_propensity, 0.0, 1.0))
        self.labor_supply = float(max(labor_supply, 0.0))
        self.skill_level = float(max(skill_level, 0.01))
        self.employed = employed
        self.wage = 0.0
        self.savings_rate = 1.0 - self.consumption_propensity
        self._rng = rng if rng is not None else np.random.default_rng()

        # Policy-specific state tracking
        self.tax_paid = 0.0
        self.subsidy_received = 0.0
        self.ubi_received = 0.0
        self.disposable_income = 0.0
        self.consumption = 0.0
        self.savings = 0.0

    def work(self, wage_rate: float) -> float:
        """
        Earn income based on wage rate, labor supply, and skill level.

        Parameters
        ----------
        wage_rate : float
            Market wage per efficiency unit.

        Returns
        -------
        float
            Gross income earned this period.
        """
        if not self.employed:
            self.income = self._unemployment_benefit()
            self.wage = 0.0
            return self.income

        self.wage = wage_rate
        # Income = wage * skill * labor_supply, with small stochastic variation
        productivity_shock = self._rng.normal(1.0, 0.05)
        self.income = (
            wage_rate * self.skill_level * self.labor_supply * max(productivity_shock, 0.1)
        )
        return self.income

    def consume(self, price_level: float = 1.0) -> float:
        """
        Determine consumption expenditure.

        Consumption is a function of disposable income and consumption propensity,
        adjusted by the price level.

        Parameters
        ----------
        price_level : float
            Current goods price level (default 1.0).

        Returns
        -------
        float
            Consumption expenditure.
        """
        # Consumption = propensity * disposable income, bounded by wealth+income
        desired = self.consumption_propensity * self.disposable_income / price_level
        budget_constraint = max(self.wealth + self.income, 0.0)
        self.consumption = np.clip(desired, 0.0, budget_constraint)
        return self.consumption

    def save(self) -> float:
        """
        Save remaining disposable income after consumption.

        Updates wealth by adding savings (disposable income minus consumption).

        Returns
        -------
        float
            Amount saved this period.
        """
        self.savings = max(self.disposable_income - self.consumption, 0.0)
        self.wealth += self.savings
        return self.savings

    def respond_to_policy(self, policy: Any) -> Dict[str, Any]:
        """
        Adjust behavior in response to a policy intervention.

        This method is called by the simulation engine when a policy is applied.
        It delegates to the policy's apply method and can trigger behavioral
        responses such as changes in labor supply or consumption propensity.

        Parameters
        ----------
        policy : Policy
            Policy object with an apply method.

        Returns
        -------
        dict
            Changes in agent state caused by the policy.
        """
        state_before = {
            "wealth": self.wealth,
            "disposable_income": self.disposable_income,
            "labor_supply": self.labor_supply,
            "consumption_propensity": self.consumption_propensity,
        }

        # Apply the policy to this agent
        policy.apply(self)

        # Behavioral responses to policy changes
        self._adjust_behavior(policy)

        state_after = {
            "wealth": self.wealth,
            "disposable_income": self.disposable_income,
            "labor_supply": self.labor_supply,
            "consumption_propensity": self.consumption_propensity,
        }

        return {k: state_after[k] - state_before[k] for k in state_before}

    def _adjust_behavior(self, policy: Any) -> None:
        """
        Internal method for behavioral response to policy.

        Agents may adjust labor supply or consumption in response to tax changes,
        subsidies, or UBI. This implements basic behavioral micro-foundations.

        Parameters
        ----------
        policy : Policy
            The applied policy.
        """
        from policysim.policy import TaxPolicy, UBiPolicy, SubsidyPolicy

        if isinstance(policy, TaxPolicy):
            # Higher taxes may reduce labor supply (substitution effect)
            effective_rate = self.tax_paid / max(self.income, 1.0)
            self.labor_supply *= max(1.0 - 0.15 * effective_rate, 0.5)

        elif isinstance(policy, UBiPolicy):
            # UBI may slightly reduce labor supply (income effect)
            self.labor_supply *= 0.98

        elif isinstance(policy, SubsidyPolicy):
            # Subsidies may increase consumption propensity for targeted goods
            expected = getattr(policy, "target_group", None)
            if expected is not None and self.skill_level < 1.0:
                self.consumption_propensity = min(self.consumption_propensity * 1.05, 1.0)

    def _unemployment_benefit(self) -> float:
        """Compute unemployment benefit (proportional to skill level)."""
        return max(self.skill_level * 100.0, 50.0)

    def to_dict(self) -> Dict[str, float]:
        """Export agent state as a dictionary."""
        return {
            "wealth": self.wealth,
            "income": self.income,
            "consumption_propensity": self.consumption_propensity,
            "labor_supply": self.labor_supply,
            "skill_level": self.skill_level,
            "employed": float(self.employed),
            "wage": self.wage,
            "tax_paid": self.tax_paid,
            "subsidy_received": self.subsidy_received,
            "ubi_received": self.ubi_received,
            "disposable_income": self.disposable_income,
            "consumption": self.consumption,
            "savings": self.savings,
        }

    def __repr__(self) -> str:
        return (
            f"EconomicAgent(wealth={self.wealth:.1f}, income={self.income:.1f}, "
            f"mpc={self.consumption_propensity:.2f}, skill={self.skill_level:.2f}, "
            f"employed={self.employed})"
        )


def create_agent_population(
    n_agents: int,
    wealth_mean: float = 5000.0,
    wealth_std: float = 3000.0,
    mpc_mean: float = 0.75,
    mpc_std: float = 0.15,
    skill_mean: float = 1.0,
    skill_std: float = 0.4,
    labor_supply_mean: float = 0.9,
    labor_supply_std: float = 0.2,
    employment_rate: float = 0.92,
    seed: Optional[int] = None,
) -> list:
    """
    Create a heterogeneous population of EconomicAgent instances.

    Agent attributes are drawn from log-normal or normal distributions
    to reflect realistic heterogeneity in the population.

    Parameters
    ----------
    n_agents : int
        Number of agents to create.
    wealth_mean : float
        Mean initial wealth (log-normal).
    wealth_std : float
        Std deviation of initial wealth (log-normal).
    mpc_mean : float
        Mean marginal propensity to consume.
    mpc_std : float
        Std deviation of MPC.
    skill_mean : float
        Mean skill level.
    skill_std : float
        Std deviation of skill level.
    labor_supply_mean : float
        Mean labor supply.
    labor_supply_std : float
        Std deviation of labor supply.
    employment_rate : float
        Fraction of agents initially employed.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of EconomicAgent
        Created agent population.
    """
    rng = np.random.default_rng(seed)

    # Wealth: log-normal distribution (positive skewed)
    wealth = rng.lognormal(
        mean=np.log(wealth_mean) - 0.5 * np.log(1 + (wealth_std / wealth_mean) ** 2),
        sigma=np.sqrt(np.log(1 + (wealth_std / wealth_mean) ** 2)),
        size=n_agents,
    )

    # MPC: beta-like via truncated normal
    mpc = np.clip(rng.normal(mpc_mean, mpc_std, size=n_agents), 0.1, 0.99)

    # Skill: log-normal
    skill = rng.lognormal(
        mean=np.log(skill_mean) - 0.5 * np.log(1 + (skill_std / skill_mean) ** 2),
        sigma=np.sqrt(np.log(1 + (skill_std / skill_mean) ** 2)),
        size=n_agents,
    )

    # Labor supply: truncated normal
    labor = np.clip(rng.normal(labor_supply_mean, labor_supply_std, size=n_agents), 0.1, 1.5)

    # Employment: Bernoulli
    employed = rng.random(size=n_agents) < employment_rate

    agents = []
    for i in range(n_agents):
        agent_rng = np.random.default_rng(seed + i + 1 if seed is not None else None)
        agents.append(
            EconomicAgent(
                wealth=wealth[i],
                consumption_propensity=mpc[i],
                skill_level=skill[i],
                labor_supply=labor[i],
                employed=bool(employed[i]),
                rng=agent_rng,
            )
        )

    return agents
