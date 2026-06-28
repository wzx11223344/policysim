"""
Counterfactual analysis for PolicySim.

Implements Difference-in-Differences (DiD) estimation, Synthetic Control
Method, placebo/permutation tests, and event study visualization for
causal policy evaluation.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import warnings


def did_estimate(
    pre_data: np.ndarray,
    post_data: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
) -> Dict[str, float]:
    """
    Difference-in-Differences (DiD) estimator.

    Estimates the Average Treatment Effect on the Treated (ATT) using the
    classic DiD specification:

        ATT = (T_post - T_pre) - (C_post - C_pre)

    Parameters
    ----------
    pre_data : np.ndarray
        Outcome values for all units in the pre-treatment period.
    post_data : np.ndarray
        Outcome values for all units in the post-treatment period.
    treated : np.ndarray
        Boolean mask or indices for treated units.
    control : np.ndarray
        Boolean mask or indices for control units.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'att': Average Treatment Effect on the Treated
        - 'treated_pre_mean': Mean pre-treatment outcome for treated
        - 'treated_post_mean': Mean post-treatment outcome for treated
        - 'control_pre_mean': Mean pre-treatment outcome for control
        - 'control_post_mean': Mean post-treatment outcome for control
        - 'counterfactual': What treated post would have been without treatment
        - 'standard_error': Standard error of ATT (conservative estimate)

    Notes
    -----
    The parallel trends assumption is required for valid DiD estimation.
    This implementation computes a conservative standard error based on
    the variance of the pre-treatment differences.
    """
    treated = np.asarray(treated, dtype=bool)
    control = np.asarray(control, dtype=bool)

    # Ensure no overlap
    if np.any(treated & control):
        raise ValueError("Treated and control units must be disjoint.")

    # Compute means
    t_pre = np.mean(pre_data[treated])
    t_post = np.mean(post_data[treated])
    c_pre = np.mean(pre_data[control])
    c_post = np.mean(post_data[control])

    # DiD estimator
    att = (t_post - t_pre) - (c_post - c_pre)

    # Counterfactual: what treated post would have been without treatment
    counterfactual = t_pre + (c_post - c_pre)

    # Conservative standard error (using pre-treatment differences)
    treated_diff = post_data[treated] - pre_data[treated]
    control_diff = post_data[control] - pre_data[control]
    pooled_var = (np.var(treated_diff, ddof=1) + np.var(control_diff, ddof=1)) / 2.0
    n_t = treated.sum()
    n_c = control.sum()
    se = np.sqrt(pooled_var * (1.0 / n_t + 1.0 / n_c)) if n_t > 1 and n_c > 1 else np.nan

    return {
        "att": float(att),
        "treated_pre_mean": float(t_pre),
        "treated_post_mean": float(t_post),
        "control_pre_mean": float(c_pre),
        "control_post_mean": float(c_post),
        "counterfactual": float(counterfactual),
        "standard_error": float(se),
    }


def synthetic_control(
    treated_unit: np.ndarray,
    control_pool: np.ndarray,
    pre_periods: int,
    method: str = "ols",
) -> Dict[str, Any]:
    """
    Synthetic Control Method for counterfactual estimation.

    Constructs a synthetic control unit as a weighted combination of control
    units such that the synthetic unit closely matches the treated unit's
    pre-treatment outcomes.

    Parameters
    ----------
    treated_unit : np.ndarray
        Time series of outcomes for the treated unit.
        Shape: (n_periods,)
    control_pool : np.ndarray
        Time series of outcomes for control units.
        Shape: (n_controls, n_periods)
    pre_periods : int
        Number of pre-treatment periods.
    method : str
        Weight estimation method. Options:
        - 'ols': Ordinary least squares (default)
        - 'constrained': Constrained OLS (weights >= 0, sum to 1)
        - 'elastic_net': Elastic net regularization

    Returns
    -------
    dict
        Dictionary with keys:
        - 'weights': Estimated weights for each control unit
        - 'synthetic_pre': Synthetic unit pre-treatment trajectory
        - 'synthetic_post': Synthetic unit post-treatment trajectory (counterfactual)
        - 'att': Average Treatment Effect (post-treatment)
        - 'att_time': Period-by-period treatment effects
        - 'pre_fit_rmse': Root mean squared error in pre-period
    """
    treated_unit = np.asarray(treated_unit)
    control_pool = np.asarray(control_pool)

    n_periods = len(treated_unit)
    n_controls = control_pool.shape[0]

    if n_controls < 2:
        raise ValueError("Need at least 2 control units for synthetic control.")

    # Split into pre and post
    X = control_pool[:, :pre_periods].T  # (pre_periods, n_controls)
    y = treated_unit[:pre_periods]  # (pre_periods,)

    if method == "ols":
        weights = np.linalg.lstsq(X, y, rcond=None)[0]
    elif method == "constrained":
        weights = _constrained_ols(X, y)
    elif method == "elastic_net":
        weights = _elastic_net_weights(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Synthetic unit trajectory
    synthetic = control_pool.T @ weights  # (n_periods,)

    # Pre-fit quality
    pre_rmse = np.sqrt(np.mean((synthetic[:pre_periods] - treated_unit[:pre_periods]) ** 2))

    # Treatment effects
    att_time = treated_unit[pre_periods:] - synthetic[pre_periods:]
    att = np.mean(att_time)

    return {
        "weights": weights,
        "synthetic_pre": synthetic[:pre_periods],
        "synthetic_post": synthetic[pre_periods:],
        "att": float(att),
        "att_time": att_time,
        "pre_fit_rmse": float(pre_rmse),
    }


def _constrained_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constrained OLS: weights >= 0, sum to 1.

    Uses quadratic programming via SciPy.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (pre_periods, n_controls).
    y : np.ndarray
        Target vector (pre_periods,).

    Returns
    -------
    np.ndarray
        Constrained weights.
    """
    try:
        from scipy.optimize import minimize

        n_controls = X.shape[1]

        def objective(w):
            residuals = y - X @ w
            return np.sum(residuals**2)

        # Constraints: sum(w) = 1, w >= 0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_controls)]

        result = minimize(
            objective,
            x0=np.ones(n_controls) / n_controls,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        return result.x
    except ImportError:
        warnings.warn("SciPy not available, falling back to OLS. Install scipy for constrained optimization.")
        return np.linalg.lstsq(X, y, rcond=None)[0]


def _elastic_net_weights(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, l1_ratio: float = 0.5) -> np.ndarray:
    """
    Elastic Net regularized weights.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray
        Target vector.
    alpha : float
        Regularization strength.
    l1_ratio : float
        L1 ratio (0 = ridge, 1 = lasso).

    Returns
    -------
    np.ndarray
        Regularized weights.
    """
    n_controls = X.shape[1]

    # Use iterative soft-thresholding (coordinate descent)
    weights = np.linalg.lstsq(X, y, rcond=None)[0]
    weights = np.clip(weights, 0, None)

    for _ in range(100):
        for j in range(n_controls):
            # Compute partial residual
            r = y - X @ weights + X[:, j] * weights[j]
            rho = X[:, j] @ r
            # Soft threshold
            if rho < -alpha * l1_ratio:
                weights[j] = (rho + alpha * l1_ratio) / (X[:, j] @ X[:, j] + alpha * (1 - l1_ratio))
            elif rho > alpha * l1_ratio:
                weights[j] = (rho - alpha * l1_ratio) / (X[:, j] @ X[:, j] + alpha * (1 - l1_ratio))
            else:
                weights[j] = 0.0

    return weights


def placebo_test(
    data: np.ndarray,
    treatment_period: int,
    n_placebos: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Placebo/permutation test for DiD significance.

    Randomly assigns treatment timing to estimate the distribution of
    treatment effects under the null hypothesis (no effect).

    Parameters
    ----------
    data : np.ndarray
        Panel data. Shape: (n_units, n_periods)
    treatment_period : int
        Period at which treatment occurs.
    n_placebos : int
        Number of placebo draws.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'observed_effect': The actual observed treatment effect
        - 'placebo_effects': Array of placebo effects
        - 'p_value': Proportion of placebos with absolute effect >= observed
        - 'ci_95': 95% confidence interval of placebo distribution
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n_units, n_periods = data.shape

    placebo_effects = np.zeros(n_placebos)

    for i in range(n_placebos):
        # Randomly assign half as "treated", half as "control"
        permuted = rng.permutation(n_units)
        n_treated = n_units // 2
        treated_idx = permuted[:n_treated]
        control_idx = permuted[n_treated:]

        t_pre = np.mean(data[treated_idx, :treatment_period])
        t_post = np.mean(data[treated_idx, treatment_period:])
        c_pre = np.mean(data[control_idx, :treatment_period])
        c_post = np.mean(data[control_idx, treatment_period:])

        placebo_effects[i] = (t_post - t_pre) - (c_post - c_pre)

    # Observed effect: treat all units the same (conservative)
    observed = np.mean(data[:, treatment_period:]) - np.mean(data[:, :treatment_period])

    # Two-sided p-value
    p_value = np.mean(np.abs(placebo_effects) >= np.abs(observed))

    # 95% CI
    ci_low = np.percentile(placebo_effects, 2.5)
    ci_high = np.percentile(placebo_effects, 97.5)

    return {
        "observed_effect": float(observed),
        "placebo_effects": placebo_effects,
        "p_value": float(p_value),
        "ci_95": (float(ci_low), float(ci_high)),
    }


def event_study(
    data: np.ndarray,
    treatment_period: int,
    leads: int = 5,
    lags: int = 10,
    treated_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Event study analysis for dynamic treatment effects.

    Estimates treatment effects at each lead/lag relative to the treatment
    period. Useful for testing parallel trends (leads should be ~0) and
    visualizing dynamic effects (lags show treatment evolution).

    Parameters
    ----------
    data : np.ndarray
        Panel data. Shape: (n_units, n_periods)
    treatment_period : int
        Period at which treatment occurs (0-indexed).
    leads : int
        Number of pre-treatment leads to estimate.
    lags : int
        Number of post-treatment lags to estimate.
    treated_mask : np.ndarray, optional
        Boolean array indicating treated units. If None, assumes all units
        are treated.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'coefficients': Array of event-time coefficients (length = leads + lags + 1)
        - 'periods': Array of relative period indices (-leads to +lags)
        - 'se': Standard errors of coefficients (if estimable)
    """
    data = np.asarray(data)
    n_units, n_periods = data.shape

    if treated_mask is None:
        treated_mask = np.ones(n_units, dtype=bool)

    treated_mask = np.asarray(treated_mask, dtype=bool)
    control_mask = ~treated_mask

    # For each relative period, compute treated - control difference
    total_periods = leads + lags + 1
    coefficients = np.zeros(total_periods)
    periods = np.arange(-leads, lags + 1)

    for i, rel_period in enumerate(periods):
        abs_period = treatment_period + rel_period
        if 0 <= abs_period < n_periods:
            treated_mean = np.mean(data[treated_mask, abs_period])
            control_mean = np.mean(data[control_mask, abs_period]) if control_mask.sum() > 0 else 0.0
            coefficients[i] = treated_mean - control_mean
        else:
            coefficients[i] = np.nan

    # Simple standard error estimate (using pre-period variance)
    if control_mask.sum() > 0:
        pre_diffs = data[treated_mask, :treatment_period].mean(axis=0) - data[control_mask, :treatment_period].mean(axis=0)
        se = np.ones(total_periods) * np.std(pre_diffs, ddof=1) if len(pre_diffs) > 1 else np.ones(total_periods) * np.nan
    else:
        se = np.ones(total_periods) * np.nan

    return {
        "coefficients": coefficients,
        "periods": periods,
        "se": se,
    }


def plot_event_study(
    event_study_result: Dict[str, np.ndarray],
    ax=None,
    title: str = "Event Study",
    xlabel: str = "Periods Relative to Treatment",
    ylabel: str = "Treatment Effect",
    color: str = "#2196F3",
) -> Any:
    """
    Plot event study coefficients with confidence bands.

    Parameters
    ----------
    event_study_result : dict
        Output from event_study().
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    color : str
        Line color.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    periods = event_study_result["periods"]
    coefs = event_study_result["coefficients"]
    se = event_study_result["se"]

    # Plot coefficients
    ax.plot(periods, coefs, color=color, linewidth=1.5, marker="o", markersize=4)
    ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.7, label="Treatment")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Confidence bands
    if not np.all(np.isnan(se)):
        ax.fill_between(
            periods,
            coefs - 1.96 * se,
            coefs + 1.96 * se,
            color=color,
            alpha=0.15,
            label="95% CI",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def parallel_trends_test(
    pre_data_treated: np.ndarray,
    pre_data_control: np.ndarray,
) -> Dict[str, float]:
    """
    Test the parallel trends assumption for DiD.

    Compares pre-treatment trends between treated and control groups.
    A significant difference suggests violation of parallel trends.

    Parameters
    ----------
    pre_data_treated : np.ndarray
        Pre-treatment data for treated group. Shape: (n_treated, n_pre_periods)
    pre_data_control : np.ndarray
        Pre-treatment data for control group. Shape: (n_control, n_pre_periods)

    Returns
    -------
    dict
        Dictionary with:
        - 'treated_trend': Mean period-over-period change for treated
        - 'control_trend': Mean period-over-period change for control
        - 'trend_difference': Difference in trends
        - 't_statistic': t-statistic for trend difference (if computable)
        - 'parallel_trends_holds': Whether trends are statistically similar (|t| < 2)
    """
    # Compute period-over-period changes
    treated_diffs = np.diff(pre_data_treated, axis=1).mean(axis=1)
    control_diffs = np.diff(pre_data_control, axis=1).mean(axis=1)

    treated_trend = float(np.mean(treated_diffs))
    control_trend = float(np.mean(control_diffs))
    trend_diff = treated_trend - control_trend

    # Pooled standard error
    n_t = len(treated_diffs)
    n_c = len(control_diffs)
    pooled_var = ((n_t - 1) * np.var(treated_diffs, ddof=1) + (n_c - 1) * np.var(control_diffs, ddof=1)) / (n_t + n_c - 2)
    se_diff = np.sqrt(pooled_var * (1.0 / n_t + 1.0 / n_c)) if n_t > 1 and n_c > 1 else np.inf

    t_stat = trend_diff / se_diff if se_diff > 0 else 0.0
    holds = abs(t_stat) < 2.0

    return {
        "treated_trend": treated_trend,
        "control_trend": control_trend,
        "trend_difference": trend_diff,
        "t_statistic": float(t_stat),
        "parallel_trends_holds": holds,
    }
