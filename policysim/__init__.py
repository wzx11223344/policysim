"""
PolicySim: Policy Simulation & Counterfactual Analysis Engine.

Combines Agent-Based Modeling (ABM), microsimulation, and
DiD/Synthetic Control counterfactuals for policy analysis.
"""

__version__ = "0.2.0"
__author__ = "PolicySim Contributors"

from policysim.agents import EconomicAgent
from policysim.market import clear_labor_market, clear_goods_market
from policysim.policy import (
    TaxPolicy,
    SubsidyPolicy,
    UBiPolicy,
    MinimumWagePolicy,
    InterestRatePolicy,
)
from policysim.abm import Simulation

__all__ = [
    "EconomicAgent",
    "clear_labor_market",
    "clear_goods_market",
    "TaxPolicy",
    "SubsidyPolicy",
    "UBiPolicy",
    "MinimumWagePolicy",
    "InterestRatePolicy",
    "Simulation",
]
