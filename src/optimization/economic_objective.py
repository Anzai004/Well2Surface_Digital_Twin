"""
CSS Cycle Economic Objective & Production-Cutoff Trigger Module.

Implements the lifecycle profit objective J for a single CSS (Huff-Soak-Puff)
cycle, the instantaneous marginal profit rate dJ/dt, the cumulative
Steam-Oil Ratio (SOR), and the production-phase cutoff trigger (PRD FR-4.3)
that signals when the current Puff (production) phase should end and the
next Huff (steam injection) cycle should begin.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ArrayOrFloat = Union[float, np.ndarray]


def calculate_instantaneous_profit_rate(
    q_o: ArrayOrFloat,
    P_motor: ArrayOrFloat,
    sigma_rod: ArrayOrFloat,
    P_oil: float,
    C_elec: float,
    C_wear: float,
) -> ArrayOrFloat:
    """
    Computes the instantaneous marginal profit rate dJ/dt during production.

    Formula:
        dJ/dt = P_oil * q_o(t) - C_elec * P_motor(t) - C_wear * sigma_rod(t)

    Args:
        q_o: Oil production rate at each instant (e.g. bbl/day). Scalar or
            array. Must be non-negative.
        P_motor: Pump motor electrical power draw at each instant (e.g. kW).
            Scalar or array. Must be non-negative.
        sigma_rod: Peak rod stress at each instant (e.g. psi), used as a
            proxy for cumulative mechanical wear. Scalar or array. Must be
            non-negative.
        P_oil: Oil sale price per unit produced (e.g. USD/bbl). Must be
            non-negative.
        C_elec: Electricity cost per unit motor power (e.g. USD/kWh). Must
            be non-negative.
        C_wear: Equipment wear cost coefficient per unit rod stress. Must be
            non-negative.

    Returns:
        float or np.ndarray: dJ/dt at each supplied instant, same shape as
        the input arrays (broadcast rules apply).

    Raises:
        ValueError: If any of q_o, P_motor, sigma_rod is negative anywhere,
            or if P_oil, C_elec, C_wear is negative.
    """
    q_o_arr = np.asarray(q_o, dtype=float)
    P_motor_arr = np.asarray(P_motor, dtype=float)
    sigma_rod_arr = np.asarray(sigma_rod, dtype=float)

    if np.any(q_o_arr < 0):
        raise ValueError("Oil production rate q_o cannot be negative.")
    if np.any(P_motor_arr < 0):
        raise ValueError("Motor power P_motor cannot be negative.")
    if np.any(sigma_rod_arr < 0):
        raise ValueError("Rod stress sigma_rod cannot be negative.")
    if P_oil < 0 or C_elec < 0 or C_wear < 0:
        raise ValueError("Economic coefficients (P_oil, C_elec, C_wear) must be non-negative.")

    profit_rate = P_oil * q_o_arr - C_elec * P_motor_arr - C_wear * sigma_rod_arr

    is_scalar_input = (
        np.isscalar(q_o) and np.isscalar(P_motor) and np.isscalar(sigma_rod)
    )
    if is_scalar_input:
        return float(profit_rate)
    return profit_rate


def calculate_steam_oil_ratio(
    steam_volume: ArrayOrFloat,
    oil_volume: ArrayOrFloat,
) -> ArrayOrFloat:
    """
    Computes the Steam-Oil Ratio SOR = steam_volume / oil_volume.

    Formula:
        SOR = V_steam / V_oil

    Note:
        At nodes where oil_volume is exactly zero (e.g. t=0, before any
        production has accumulated), SOR is returned as +inf rather than
        raising -- this is the conventional reading of "infinitely poor"
        steam utilization before first oil, and lets cumulative_sor arrays
        be evaluated safely from the very first time node.

    Args:
        steam_volume: Cumulative steam volume injected (e.g. m^3), scalar or
            array. Must be non-negative.
        oil_volume: Cumulative oil volume produced (e.g. m^3), scalar or
            array. Must be non-negative.

    Returns:
        float or np.ndarray: SOR at each supplied node (dimensionless).

    Raises:
        ValueError: If steam_volume or oil_volume is negative anywhere.
    """
    is_scalar_input = np.isscalar(steam_volume) and np.isscalar(oil_volume)

    steam_arr = np.atleast_1d(np.asarray(steam_volume, dtype=float))
    oil_arr = np.atleast_1d(np.asarray(oil_volume, dtype=float))

    if np.any(steam_arr < 0):
        raise ValueError("steam_volume cannot be negative.")
    if np.any(oil_arr < 0):
        raise ValueError("oil_volume cannot be negative.")

    with np.errstate(divide="ignore", invalid="ignore"):
        sor = np.where(oil_arr > 0.0, steam_arr / np.where(oil_arr > 0.0, oil_arr, 1.0), np.inf)

    if is_scalar_input:
        return float(sor[0])
    return sor


@dataclass
class EconomicCycleResult:
    """
    Container for a full CSS production-cycle economic evaluation.

    Attributes:
        t (np.ndarray): Time nodes (days), as supplied.
        profit_rate (np.ndarray): Instantaneous marginal profit rate dJ/dt at
            each node.
        cumulative_profit (np.ndarray): Running trapezoidal integral of
            profit_rate over t, i.e. integral_0^t (dJ/dt) dt -- this is
            BEFORE subtracting the upfront lump-sum steam cost.
        J_total (float): Full-cycle net profit,
            J = cumulative_profit[-1] - C_steam * m_s.
        cumulative_oil_volume (np.ndarray): Running trapezoidal integral of
            q_o over t (cumulative oil produced).
        cumulative_sor (np.ndarray): Running cumulative Steam-Oil Ratio at
            each node (see calculate_steam_oil_ratio).
        cutoff_triggered (bool): True if a cutoff condition was met anywhere
            in the series.
        cutoff_index (Optional[int]): Index of the first node at which a
            cutoff condition was met, or None if never triggered.
        cutoff_time (Optional[float]): t value at cutoff_index, or None.
        cutoff_reason (Optional[str]): One of "profit_rate", "sor_threshold",
            "profit_rate_and_sor_threshold", or None.
    """

    t: np.ndarray
    profit_rate: np.ndarray
    cumulative_profit: np.ndarray
    J_total: float
    cumulative_oil_volume: np.ndarray
    cumulative_sor: np.ndarray
    cutoff_triggered: bool
    cutoff_index: Optional[int]
    cutoff_time: Optional[float]
    cutoff_reason: Optional[str]


def evaluate_css_cycle_economics(
    t: np.ndarray,
    q_o: np.ndarray,
    P_motor: np.ndarray,
    sigma_rod: np.ndarray,
    steam_volume_cumulative: np.ndarray,
    m_s: float,
    P_oil: float,
    C_elec: float,
    C_wear: float,
    C_steam: float,
    sor_threshold: Optional[float] = None,
) -> EconomicCycleResult:
    """
    Evaluates the full-cycle economic objective J and the FR-4.3 cutoff
    trigger over a production (Puff) time series.

    Formula:
        J = integral_0^t_cycle [P_oil*q_o(t) - C_elec*P_motor(t)
                                  - C_wear*sigma_rod(t)] dt  -  C_steam * m_s

    Cutoff trigger (PRD FR-4.3): the first time node t* at which EITHER
        dJ/dt(t*) <= 0                                  (marginal loss), OR
        cumulative_SOR(t*) > sor_threshold               (uneconomic SOR,
                                                           only checked if
                                                           sor_threshold is
                                                           supplied)
    is flagged as the recommended end of the Puff phase / start of the next
    Huff cycle.

    Args:
        t: 1-D strictly increasing array of time nodes (days), length >= 1.
        q_o: Oil production rate at each node in t (e.g. bbl/day). Same
            shape as t. Must be non-negative.
        P_motor: Pump motor power draw at each node in t (e.g. kW). Same
            shape as t. Must be non-negative.
        sigma_rod: Peak rod stress at each node in t (e.g. psi). Same shape
            as t. Must be non-negative.
        steam_volume_cumulative: Cumulative steam volume injected up to and
            including each node in t (e.g. m^3). Same shape as t. Must be
            non-negative and non-decreasing.
        m_s: Total steam mass injected during the preceding Huff phase
            (kg). Must be non-negative.
        P_oil: Oil sale price per unit produced (e.g. USD/bbl).
        C_elec: Electricity cost per unit motor power (e.g. USD/kWh).
        C_wear: Equipment wear cost coefficient per unit rod stress.
        C_steam: Upfront steam cost per unit steam mass (e.g. USD/kg).
        sor_threshold: Optional economic SOR ceiling. If cumulative SOR
            exceeds this value, the cutoff trigger fires even if dJ/dt is
            still positive. If None, only the dJ/dt <= 0 condition is
            checked.

    Returns:
        EconomicCycleResult: Full time-series economics and cutoff
        diagnostics for this cycle.

    Raises:
        ValueError: If t is empty, not 1-D, or not strictly increasing; if
            any input array does not match the shape of t; if any rate,
            stress, or cost input is negative; if steam_volume_cumulative is
            not non-decreasing; or if sor_threshold is supplied and is not
            strictly positive.
    """
    t_arr = np.asarray(t, dtype=float)
    q_o_arr = np.asarray(q_o, dtype=float)
    P_motor_arr = np.asarray(P_motor, dtype=float)
    sigma_rod_arr = np.asarray(sigma_rod, dtype=float)
    steam_vol_arr = np.asarray(steam_volume_cumulative, dtype=float)

    if t_arr.ndim != 1 or t_arr.shape[0] == 0:
        raise ValueError("t must be a non-empty 1-D array.")

    n = t_arr.shape[0]
    for name, arr in (
        ("q_o", q_o_arr),
        ("P_motor", P_motor_arr),
        ("sigma_rod", sigma_rod_arr),
        ("steam_volume_cumulative", steam_vol_arr),
    ):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have the same shape as t {(n,)}, got {arr.shape}.")

    if n > 1 and np.any(np.diff(t_arr) <= 0):
        raise ValueError("t must be strictly increasing (non-monotonic time array).")

    if np.any(q_o_arr < 0):
        raise ValueError("Oil production rate q_o cannot be negative.")
    if np.any(P_motor_arr < 0):
        raise ValueError("Motor power P_motor cannot be negative.")
    if np.any(sigma_rod_arr < 0):
        raise ValueError("Rod stress sigma_rod cannot be negative.")
    if np.any(steam_vol_arr < 0):
        raise ValueError("steam_volume_cumulative cannot be negative.")
    if n > 1 and np.any(np.diff(steam_vol_arr) < 0):
        raise ValueError("steam_volume_cumulative must be non-decreasing (it is cumulative).")
    if m_s < 0:
        raise ValueError(f"Steam mass m_s ({m_s}) cannot be negative.")
    if P_oil < 0 or C_elec < 0 or C_wear < 0 or C_steam < 0:
        raise ValueError("Economic coefficients (P_oil, C_elec, C_wear, C_steam) must be non-negative.")
    if sor_threshold is not None and sor_threshold <= 0:
        raise ValueError(f"sor_threshold ({sor_threshold}) must be strictly positive if provided.")

    profit_rate = calculate_instantaneous_profit_rate(
        q_o_arr, P_motor_arr, sigma_rod_arr, P_oil, C_elec, C_wear
    )

    if n == 1:
        cumulative_profit = np.array([0.0])
        cumulative_oil_volume = np.array([0.0])
    else:
        dt = np.diff(t_arr)
        cumulative_profit = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (profit_rate[:-1] + profit_rate[1:]) * dt),
        ))
        cumulative_oil_volume = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (q_o_arr[:-1] + q_o_arr[1:]) * dt),
        ))

    J_total = float(cumulative_profit[-1] - C_steam * m_s)

    cumulative_sor = calculate_steam_oil_ratio(steam_vol_arr, cumulative_oil_volume)
    cumulative_sor = np.atleast_1d(cumulative_sor)

    profit_cutoff_mask = profit_rate <= 0.0
    if sor_threshold is not None:
        sor_cutoff_mask = cumulative_sor > sor_threshold
    else:
        sor_cutoff_mask = np.zeros(n, dtype=bool)

    combined_mask = profit_cutoff_mask | sor_cutoff_mask

    cutoff_triggered = bool(np.any(combined_mask))
    cutoff_index: Optional[int] = None
    cutoff_time: Optional[float] = None
    cutoff_reason: Optional[str] = None

    if cutoff_triggered:
        cutoff_index = int(np.argmax(combined_mask))  # first True
        cutoff_time = float(t_arr[cutoff_index])
        profit_fired = bool(profit_cutoff_mask[cutoff_index])
        sor_fired = bool(sor_cutoff_mask[cutoff_index])
        if profit_fired and sor_fired:
            cutoff_reason = "profit_rate_and_sor_threshold"
        elif profit_fired:
            cutoff_reason = "profit_rate"
        else:
            cutoff_reason = "sor_threshold"

    return EconomicCycleResult(
        t=t_arr,
        profit_rate=profit_rate,
        cumulative_profit=cumulative_profit,
        J_total=J_total,
        cumulative_oil_volume=cumulative_oil_volume,
        cumulative_sor=cumulative_sor,
        cutoff_triggered=cutoff_triggered,
        cutoff_index=cutoff_index,
        cutoff_time=cutoff_time,
        cutoff_reason=cutoff_reason,
    )