"""
Market clearing mechanisms for PolicySim.

Implements labor market and goods market equilibrium using
Walrasian tatonnement-style price/wage discovery.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def clear_labor_market(
    agents: List[Any],
    firms_demand: Optional[callable] = None,
    min_wage: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> float:
    """
    Find equilibrium wage via iterative market clearing.

    Uses a tatonnement process: adjust wage up when labor demand exceeds supply,
    adjust down when supply exceeds demand. Returns equilibrium wage.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population supplying labor.
    firms_demand : callable, optional
        Function wage -> labor_demand. If None, uses a default CES demand curve.
    min_wage : float
        Minimum wage floor. Wage cannot clear below this level.
    max_iterations : int
        Maximum iterations for tatonnement process.
    tolerance : float
        Convergence tolerance for excess demand.

    Returns
    -------
    float
        Equilibrium wage per efficiency unit.
    """
    if firms_demand is None:
        firms_demand = _default_labor_demand

    # Initial wage guess based on average skill
    avg_skill = np.mean([a.skill_level for a in agents])
    wage = avg_skill * 50.0

    for iteration in range(max_iterations):
        # Aggregate labor supply at current wage
        labor_supply = _aggregate_labor_supply(agents, wage)
        labor_demand = firms_demand(wage, agents)

        excess_demand = labor_demand - labor_supply
        total_market = max(labor_supply, labor_demand, 1.0)

        if abs(excess_demand) / total_market < tolerance:
            break

        # Tatonnement adjustment (proportional to excess demand)
        adjustment = 0.15 * (excess_demand / total_market)
        wage = max(wage * (1.0 + adjustment), min_wage + 0.01)

    # Enforce minimum wage
    wage = max(wage, min_wage)

    # Update agent employment based on equilibrium
    _update_employment(agents, wage, firms_demand(wage, agents))

    return wage


def clear_goods_market(
    agents: List[Any],
    firms_supply: Optional[callable] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    initial_price: float = 1.0,
) -> float:
    """
    Find equilibrium price level via iterative goods market clearing.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population with consumption demand.
    firms_supply : callable, optional
        Function price -> goods_supply. If None, uses a default supply curve.
    max_iterations : int
        Maximum iterations for tatonnement process.
    tolerance : float
        Convergence tolerance.
    initial_price : float
        Starting price level guess.

    Returns
    -------
    float
        Equilibrium price level.
    """
    if firms_supply is None:
        firms_supply = _default_goods_supply

    price = initial_price

    for iteration in range(max_iterations):
        # Aggregate consumer demand at current price
        goods_demand = _aggregate_goods_demand(agents, price)
        goods_supply = firms_supply(price, agents)

        excess_demand = goods_demand - goods_supply
        total_market = max(goods_supply, goods_demand, 1.0)

        if abs(excess_demand) / total_market < tolerance:
            break

        # Tatonnement price adjustment
        adjustment = 0.10 * (excess_demand / total_market)
        price = max(price * (1.0 + adjustment), 0.01)

    return price


def _aggregate_labor_supply(agents: List[Any], wage: float) -> float:
    """
    Compute total labor supply at given wage.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.
    wage : float
        Current wage rate.

    Returns
    -------
    float
        Total labor supply in efficiency units.
    """
    total = 0.0
    for agent in agents:
        if agent.employed:
            # Elastic labor supply: slightly increasing with wage
            elasticity = 0.1
            supply = agent.labor_supply * agent.skill_level * (wage / 50.0) ** elasticity
            total += supply
    return total


def _default_labor_demand(wage: float, agents: List[Any]) -> float:
    """
    Default labor demand function (CES-like).

    Labor demand decreases with wage. Calibrated so that at the average
    skill level, demand roughly equals supply near the initial wage.

    Parameters
    ----------
    wage : float
        Current wage rate.
    agents : list of EconomicAgent
        Agent population (used for calibration).

    Returns
    -------
    float
        Labor demand in efficiency units.
    """
    n = len(agents)
    avg_skill = np.mean([a.skill_level for a in agents])
    # Elasticity of labor demand
    reference_wage = avg_skill * 50.0
    base_demand = n * avg_skill * 0.85
    return base_demand * (reference_wage / max(wage, 0.01)) ** 0.5


def _aggregate_goods_demand(agents: List[Any], price: float) -> float:
    """
    Compute total goods demand at given price level.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.
    price : float
        Current price level.

    Returns
    -------
    float
        Total goods demand.
    """
    total = 0.0
    for agent in agents:
        # Demand = MPC * disposable_income / price
        demand = agent.consumption_propensity * max(agent.disposable_income, 0) / max(price, 0.01)
        total += demand
    return total


def _default_goods_supply(price: float, agents: List[Any]) -> float:
    """
    Default goods supply function.

    Supply increases with price. Calibrated to total potential output.

    Parameters
    ----------
    price : float
        Current price level.
    agents : list of EconomicAgent
        Agent population.

    Returns
    -------
    float
        Goods supply.
    """
    n = len(agents)
    avg_income = np.mean([max(a.income, 100) for a in agents])
    base_output = n * avg_income * 0.8
    return base_output * (price ** 0.3)


def _update_employment(agents: List[Any], wage: float, labor_demand: float) -> None:
    """
    Update agent employment status based on labor market equilibrium.

    If labor demand is less than supply, some agents become unemployed.
    Unemployment is assigned probabilistically, biased toward lower-skill agents.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.
    wage : float
        Equilibrium wage.
    labor_demand : float
        Total labor demand at equilibrium.
    """
    labor_supply = _aggregate_labor_supply(agents, wage)

    if labor_supply <= labor_demand:
        # Full employment
        for agent in agents:
            agent.employed = True
        return

    # Unemployment: probability inversely related to skill
    employment_prob = labor_demand / max(labor_supply, 1.0)
    rng = np.random.default_rng()

    # Sort by skill for priority employment (higher skill = more likely employed)
    skill_weights = np.array([a.skill_level for a in agents])
    skill_weights = skill_weights / skill_weights.sum()

    n_employed_target = max(int(len(agents) * employment_prob), 1)
    employed_indices = set(
        rng.choice(len(agents), size=n_employed_target, replace=False, p=skill_weights)
    )

    for i, agent in enumerate(agents):
        agent.employed = i in employed_indices


def compute_gdp(agents: List[Any]) -> float:
    """
    Compute GDP from agent incomes.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.

    Returns
    -------
    float
        Total GDP (sum of all incomes).
    """
    return sum(a.income for a in agents)


def compute_unemployment_rate(agents: List[Any]) -> float:
    """
    Compute unemployment rate.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.

    Returns
    -------
    float
        Unemployment rate (fraction).
    """
    n = len(agents)
    if n == 0:
        return 0.0
    unemployed = sum(1 for a in agents if not a.employed)
    return unemployed / n


def compute_gini(agents: List[Any]) -> float:
    """
    Compute Gini coefficient of wealth distribution.

    Uses the standard formula: G = (2 * sum(i*w_i)) / (n * sum(w)) - (n+1)/n

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.

    Returns
    -------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality).
    """
    wealth = np.sort([a.wealth for a in agents])
    n = len(wealth)
    if n == 0 or wealth.sum() == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * wealth)) / (n * np.sum(wealth)) - (n + 1.0) / n
    return max(gini, 0.0)


def compute_poverty_rate(
    agents: List[Any],
    threshold: Optional[float] = None,
) -> float:
    """
    Compute poverty rate (fraction of agents below threshold).

    By default, threshold is 60% of median income — the OECD relative poverty line.

    Parameters
    ----------
    agents : list of EconomicAgent
        Agent population.
    threshold : float, optional
        Absolute poverty threshold. If None, uses relative poverty line.

    Returns
    -------
    float
        Poverty rate (fraction).
    """
    n = len(agents)
    if n == 0:
        return 0.0

    incomes = np.array([a.income + a.subsidy_received + a.ubi_received for a in agents])

    if threshold is None:
        threshold = 0.6 * np.median(incomes)

    return np.sum(incomes < threshold) / n
