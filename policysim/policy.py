"""
Policy interventions for PolicySim.

Defines tax policies, subsidies, UBI, minimum wage, and monetary policy
tools that can be applied during simulation runs.
"""

import numpy as np
from typing import List, Tuple, Optional, Any


class TaxPolicy:
    """
    Progressive or flat income tax policy.

    Supports both progressive (bracket-based) and flat tax schemes.
    A flat tax is specified by providing a single bracket.

    Parameters
    ----------
    rate : float
        Default/flat tax rate if no brackets specified.
    brackets : list of (float, float), optional
        List of (threshold, rate) tuples for progressive taxation.
        Example: [(0, 0.10), (50000, 0.25), (150000, 0.40)]
        Income below each threshold is taxed at the corresponding rate.

    Attributes
    ----------
    rate : float
        Flat tax rate (used when brackets is None).
    brackets : list or None
        Progressive tax brackets.
    total_revenue : float
        Accumulated tax revenue collected.
    """

    def __init__(
        self,
        rate: float = 0.25,
        brackets: Optional[List[Tuple[float, float]]] = None,
    ):
        self.rate = float(rate)
        self.brackets = brackets
        self.total_revenue = 0.0

    def apply(self, agent: Any) -> float:
        """
        Apply tax to a single agent.

        Computes tax liability based on the agent's income and either
        the flat rate or progressive brackets.

        Parameters
        ----------
        agent : EconomicAgent
            The agent to tax.

        Returns
        -------
        float
            Tax amount paid by the agent.
        """
        income = agent.income

        if self.brackets is not None:
            tax = self._compute_progressive_tax(income)
        else:
            tax = income * self.rate

        # Ensure non-negative tax and not exceeding income
        tax = np.clip(tax, 0.0, income)
        agent.tax_paid = tax
        agent.disposable_income = income - tax
        agent.wealth -= tax
        self.total_revenue += tax
        return tax

    def _compute_progressive_tax(self, income: float) -> float:
        """Compute tax under progressive bracket system."""
        tax = 0.0
        remaining = income
        sorted_brackets = sorted(self.brackets, key=lambda x: x[0])

        for i, (threshold, bracket_rate) in enumerate(sorted_brackets):
            next_threshold = (
                sorted_brackets[i + 1][0] if i + 1 < len(sorted_brackets) else float("inf")
            )
            bracket_width = next_threshold - threshold
            taxable_in_bracket = min(max(remaining, 0), bracket_width)
            tax += taxable_in_bracket * bracket_rate
            remaining -= bracket_width

        return tax

    def cost(self) -> float:
        """
        Total cost/revenue of the policy.

        For a tax policy, this returns negative of revenue (tax is revenue-positive).

        Returns
        -------
        float
            Total revenue collected (positive = government income).
        """
        return self.total_revenue

    def __repr__(self) -> str:
        if self.brackets:
            return f"TaxPolicy(brackets={self.brackets})"
        return f"TaxPolicy(rate={self.rate:.2%})"


class SubsidyPolicy:
    """
    Targeted subsidy policy.

    Provides cash or in-kind transfers to a specific group of agents
    defined by a targeting criterion.

    Parameters
    ----------
    amount : float
        Per-agent subsidy amount.
    target_group : str, optional
        Targeting criterion. Currently supports 'low_income' and 'unemployed'.
    target_threshold : float, optional
        Income threshold for 'low_income' targeting (percentile based).

    Attributes
    ----------
    amount : float
        Subsidy amount per recipient.
    target_group : str
        Target group identifier.
    target_threshold : float
        Threshold for targeting.
    total_cost : float
        Total subsidy expenditure.
    recipients : int
        Number of agents who received the subsidy.
    """

    def __init__(
        self,
        amount: float = 500.0,
        target_group: str = "low_income",
        target_threshold: float = 0.3,
    ):
        self.amount = float(amount)
        self.target_group = target_group
        self.target_threshold = float(target_threshold)
        self.total_cost = 0.0
        self.recipients = 0

    def apply(self, agent: Any) -> float:
        """
        Apply subsidy to an agent if they qualify.

        Parameters
        ----------
        agent : EconomicAgent
            The agent to evaluate.

        Returns
        -------
        float
            Subsidy amount received (0 if not qualified).
        """
        if self._qualifies(agent):
            agent.subsidy_received += self.amount
            agent.disposable_income += self.amount
            agent.wealth += self.amount
            self.total_cost += self.amount
            self.recipients += 1
            return self.amount
        return 0.0

    def _qualifies(self, agent: Any) -> bool:
        """Check if agent qualifies for the subsidy."""
        if self.target_group == "unemployed":
            return not agent.employed
        elif self.target_group == "low_income":
            return agent.income < agent.skill_level * 30000.0 * self.target_threshold
        elif self.target_group == "all":
            return True
        return False

    def cost(self) -> float:
        """
        Total cost of the subsidy program.

        Returns
        -------
        float
            Total expenditure.
        """
        return self.total_cost

    def __repr__(self) -> str:
        return (
            f"SubsidyPolicy(amount={self.amount:.0f}, "
            f"target='{self.target_group}', recipients={self.recipients})"
        )


class UBiPolicy:
    """
    Universal Basic Income (UBI) policy.

    Provides an unconditional fixed cash transfer to all agents
    regardless of employment or income status.

    Parameters
    ----------
    amount : float
        Per-agent UBI payment per period.
    tax_rate : float
        Flat tax rate used to finance the UBI (if applicable).

    Attributes
    ----------
    amount : float
        UBI payment amount.
    tax_rate : float
        Associated financing tax rate.
    total_cost : float
        Total UBI expenditure.
    recipients : int
        Number of recipients (should equal population size).
    """

    def __init__(self, amount: float = 1000.0, tax_rate: float = 0.20):
        self.amount = float(amount)
        self.tax_rate = float(tax_rate)
        self.total_cost = 0.0
        self.recipients = 0

    def apply(self, agent: Any) -> float:
        """
        Apply UBI payment to an agent.

        Parameters
        ----------
        agent : EconomicAgent
            The agent receiving UBI.

        Returns
        -------
        float
            UBI amount paid.
        """
        agent.ubi_received += self.amount
        agent.disposable_income += self.amount
        agent.wealth += self.amount
        self.total_cost += self.amount
        self.recipients += 1

        # UBI may be partially financed by a flat tax
        if self.tax_rate > 0:
            tax = agent.income * self.tax_rate
            agent.tax_paid += tax
            agent.disposable_income -= tax
            agent.wealth -= tax
            self.total_cost -= tax

        return self.amount

    def cost(self) -> float:
        """
        Net cost of UBI program (gross payments minus tax collected).

        Returns
        -------
        float
            Net program cost.
        """
        return self.total_cost

    def __repr__(self) -> str:
        return f"UBiPolicy(amount={self.amount:.0f}, tax_rate={self.tax_rate:.2%})"


class MinimumWagePolicy:
    """
    Minimum wage policy.

    Sets a floor on the market-clearing wage. If the equilibrium wage
    would be below the floor, the floor becomes the effective wage.

    Parameters
    ----------
    level : float
        Minimum wage per efficiency unit.
    enforce : bool
        Whether to enforce the minimum wage (affects employment).

    Attributes
    ----------
    level : float
        Minimum wage level.
    enforce : bool
        Enforcement flag.
    affected_agents : int
        Number of agents whose wage was raised by the policy.
    """

    def __init__(self, level: float = 15.0, enforce: bool = True):
        self.level = float(level)
        self.enforce = enforce
        self.affected_agents = 0

    def apply(self, wage: float) -> float:
        """
        Apply minimum wage floor to equilibrium wage.

        Parameters
        ----------
        wage : float
            Market-clearing wage.

        Returns
        -------
        float
            Effective wage after minimum wage floor.
        """
        if self.enforce and wage < self.level:
            return self.level
        return wage

    def apply_to_agent(self, agent: Any) -> None:
        """
        Ensure an individual agent's wage meets the minimum.

        Parameters
        ----------
        agent : EconomicAgent
            The agent to evaluate.
        """
        if agent.employed and agent.wage < self.level:
            agent.wage = self.level
            agent.income = max(agent.income, self.level * agent.skill_level * agent.labor_supply)
            self.affected_agents += 1

    def cost(self) -> float:
        """
        Cost of minimum wage (for government, 0 direct fiscal cost).

        Returns
        -------
        float
            Direct fiscal cost (0.0).
        """
        return 0.0

    def __repr__(self) -> str:
        return f"MinimumWagePolicy(level={self.level:.1f}, enforce={self.enforce})"


class InterestRatePolicy:
    """
    Monetary policy via interest rate.

    Affects agent savings behavior and investment returns. A higher
    rate increases returns on savings but also increases borrowing costs.

    Parameters
    ----------
    rate : float
        Annual interest rate (as decimal, e.g., 0.03 = 3%).
    transmission : float
        Transmission efficiency to real economy (0 to 1).

    Attributes
    ----------
    rate : float
        Policy interest rate.
    transmission : float
        Transmission parameter.
    """

    def __init__(self, rate: float = 0.03, transmission: float = 0.8):
        self.rate = float(rate)
        self.transmission = float(np.clip(transmission, 0.0, 1.0))

    def apply(self, agent: Any) -> float:
        """
        Apply interest rate to agent's savings.

        Savings earn (or cost) interest. Agents with positive wealth earn
        interest; those with negative wealth (debt) pay interest.

        Parameters
        ----------
        agent : EconomicAgent
            The agent affected by the interest rate.

        Returns
        -------
        float
            Interest earned (positive) or paid (negative).
        """
        effective_rate = self.rate * self.transmission
        interest = agent.wealth * effective_rate
        agent.wealth += interest
        agent.income += interest
        return interest

    def cost(self) -> float:
        """
        Fiscal cost of interest rate policy (0 direct cost).

        Returns
        -------
        float
            0.0 (monetary policy has no direct fiscal cost).
        """
        return 0.0

    def __repr__(self) -> str:
        return f"InterestRatePolicy(rate={self.rate:.2%}, transmission={self.transmission:.2f})"
