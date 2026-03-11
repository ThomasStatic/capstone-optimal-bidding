import argparse
import pandas as pd
import numpy as np
import pickle

from shell.ablations.demand_perturbation import DemandPerturbationConfig, run_demand_perturbation_sweep
from shell.ablations.run_all import AllAblationsConfig, run_all_ablations
from shell.ablations.warm_start import WarmStartAblationConfig, run_warm_start_ablation
from shell.api_controllers.market_loads_api import ISODemandController
from shell.action_space import ActionSpace
from shell.linear_approximator import (
    HIST_LOAD_COL,
    FORECAST_LOAD_COL,
    LMP_CSV_PATH,
    PRICE_COL,
    Discretizer,
)
from shell.market_model import MarketModel, MarketParams
from shell.baselines.cost_plus_markup import CostPlusMarkupPolicy
from shell.baselines.historical_quantile import HistoricalQuantilePolicy
from shell.evaluations.baseline_runner import run_policy_on_episodes
from shell.agent_interface import Observation
from shell.evaluations.policy_types import Policy
from shell.state_space import State
from shell.tabular_q_agent import TabularQLearningAgent
from shell.multi_agent_metrics import MetricsTracker, export_multi_agent_metrics
from typing import TypeAlias

ISO = "ERCOT"
START_DATE = "2025-01-01"
END_DATE = "2026-01-31"

NUM_DISCRETIZER_BINS = 8
MAX_BID_QUANTITY_MW = 400

FORECAST_CSV_PATH = "ERCOT - Load Forecast 2025.csv"
FORECAST_HORIZON_HOURS = 24 * 14  # 2 weeks

def _load_ercot_forecast_series() -> pd.Series:
    """
    Returns an hourly forecast series indexed by UTC datetime, in MW.
    Uses TOTAL_SYSTEM_LOAD columns and sums Houston+North+South+West.
    """
    df = pd.read_csv(
        FORECAST_CSV_PATH,
        na_values=["-", " -", " -   ", " -   -", "—"],
    )

    df = df.rename(columns={"Date_Time_UTC": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Pick the 4 regional total system load columns
    cols = [
        "ERCOT_HOUSTON_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_NORTH_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_SOUTH_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_WEST_TOTAL_SYSTEM_LOAD (MWh)",
    ]

    # Clean numeric (handles "13,896.77" style strings)
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df["forecast_total_mw"] = df[cols].sum(axis=1)

    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts = pd.Timestamp(END_DATE, tz="UTC")

    s = (
        df.loc[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts), ["datetime", "forecast_total_mw"]]
          .dropna()
          .set_index("datetime")["forecast_total_mw"]
          .sort_index()
    )

    return s

def apply_demand_scale_to_state(state: State, scale: float) -> None:
    if scale == 1.0:
        return
    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data not initialized")

    df = state.raw_state_data

    if HIST_LOAD_COL in df.columns:
        df[HIST_LOAD_COL] = df[HIST_LOAD_COL].astype(float) * scale

    if FORECAST_LOAD_COL in df.columns:
        df[FORECAST_LOAD_COL] = df[FORECAST_LOAD_COL].astype(float) * scale


def get_episode_temperature(ep_idx: int) -> float | None:
    """
    Returns:
      - float for fixed/exp_decay
      - None for qgap (agent computes adaptive temperature internally)
    """
    if args.temperature_mode == "fixed":
        return float(args.temperature)

    if args.temperature_mode == "qgap":
        return None

    if args.temperature_mode == "exp_decay":
        T0 = float(args.temperature)
        decay = float(args.temperature_decay)
        Tmin = float(args.temperature_min)
        return max(Tmin, T0 * (decay ** ep_idx))

    raise ValueError(f"Unknown temperature_mode: {args.temperature_mode}")

def build_state_and_discretizers(
    load_df: pd.DataFrame, price_df: pd.DataFrame
) -> tuple[State, dict]:
    load_disc = Discretizer(col=HIST_LOAD_COL, n_bins=NUM_DISCRETIZER_BINS)
    forecast_disc = Discretizer(col=FORECAST_LOAD_COL, n_bins=NUM_DISCRETIZER_BINS)
    price_disc = Discretizer(col=PRICE_COL, n_bins=NUM_DISCRETIZER_BINS)

    # plant id can be any string identifier - not currently used in logic
    state = State(
        plant_id="plant_1",
        discretizers={
            "load": load_disc,
            "forecast": forecast_disc,
            "price": price_disc,
        },
    )

    if args.verbose:
        print("LMP min/max:", price_df["datetime"].min(), price_df["datetime"].max())
        print("Load columns:", list(load_df.columns))
        ts_col = "datetime" if "datetime" in load_df.columns else "period"
        print("Load min/max:", load_df[ts_col].min(), load_df[ts_col].max())
        print("LMP tz:", price_df["datetime"].dtype)
        print("Load tz:", load_df["period"].dtype)

    state.load_state_data(
        {
            "load": load_df.rename(columns={"period": "datetime"}),
            "price": price_df,
        }
    )

    # On first pass, discretize without any forecast data
    state.apply()

    return state, {
        "load": load_disc,
        "forecast": forecast_disc,
        "price": price_disc,
    }

# When we call subsequent state.apply() during episodes, we will have forecast data
def inject_epsisode_forecast(state: State, forecast_df: pd.DataFrame) -> None:
    '''Injects 24-hour forecast data into the state for the episode'''
    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been intialized")
    
    df = state.raw_state_data

    if FORECAST_LOAD_COL not in df.columns:
        df[FORECAST_LOAD_COL] = np.nan

    forecast_df = forecast_df.copy()
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"], utc=True)

    if float(args.demand_scale) != 1.0 and FORECAST_LOAD_COL in forecast_df.columns:
        forecast_df[FORECAST_LOAD_COL] = forecast_df[FORECAST_LOAD_COL].astype(float) * float(args.demand_scale)

    f = forecast_df.set_index("datetime")[FORECAST_LOAD_COL]
    df.loc[f.index, FORECAST_LOAD_COL] = f.values

def make_action_space(lmp_df: pd.DataFrame) -> ActionSpace:
    price_disc = Discretizer(col=PRICE_COL, n_bins=NUM_DISCRETIZER_BINS)
    price_disc.fit(lmp_df[[PRICE_COL]])

    qty_grid = np.linspace(0, MAX_BID_QUANTITY_MW, 100)
    qty_df = pd.DataFrame({ "quantity": qty_grid })
    qty_disc = Discretizer(col="quantity", n_bins=NUM_DISCRETIZER_BINS)
    qty_disc.fit(qty_df)

    return ActionSpace(price_disc=price_disc, quantity_disc=qty_disc)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    lmp_df = pd.read_csv(LMP_CSV_PATH, parse_dates=["CloseDateUTC"])
    lmp_df = lmp_df.rename(columns={"CloseDateUTC": "datetime"})
    lmp_df[PRICE_COL] = pd.to_numeric(lmp_df[PRICE_COL], errors="coerce")
    lmp_df["datetime"] = pd.to_datetime(lmp_df["datetime"], utc=True)
    lmp_df = lmp_df.loc[
        (lmp_df["datetime"] >= pd.to_datetime(START_DATE, utc=True)) &
        (lmp_df["datetime"] <= pd.to_datetime(END_DATE, utc=True))
    ]

    lmp_start = lmp_df["datetime"].min()
    lmp_end   = lmp_df["datetime"].max()
    start = lmp_start.date().isoformat()
    end   = lmp_end.date().isoformat()

    historic_load_api = ISODemandController(start, end, ISO)
    load_df = historic_load_api.get_market_loads()
    load_df =load_df[load_df["respondent"] == ISO].copy()
    load_df["period"] = pd.to_datetime(load_df["period"], utc=True)
    load_df = load_df.sort_values("period")

    return load_df, lmp_df

def load_fuel_mix() -> pd.DataFrame:
    """
    Load fuel mix data at hourly resolution.
    
    Returns a DataFrame with columns:
    - datetime: UTC timestamps (hourly)
    - Total_MW: Total system load
    - Fuel type columns (Biomass, Coal, Hydro, Natural Gas, Nuclear, Other, Solar, Wind)
      containing proportion values that sum to ~1.0
    """
    fuel_mix_df = pd.read_csv("fuel_mix_2025_2026_ercot.csv")
    fuel_mix_df["datetime"] = pd.to_datetime(fuel_mix_df["datetime"], utc=True)
    fuel_mix_df = fuel_mix_df.sort_values("datetime")
    return fuel_mix_df

def load_active_generators() -> pd.DataFrame:
    """
    Load active generators data.
    
    Returns a DataFrame with columns:
    - Power_Plant_ID: Unique generator ID
    - plant_name: Generator name
    - fuel_type: Type of fuel (Coal, Natural Gas, Hydro, etc.)
    - capacity: Capacity in MW
    - capacity_factor_2025: Capacity utilization factor
    """
    generators_df = pd.read_csv("active_generators_2025_ercot.csv")
    return generators_df

def build_opponent_agents(generators_df: pd.DataFrame) -> list[dict]:
    """
    Build a list of opponent "agents" from active generators.
    
    Each agent is a dict containing:
    - id: Power_Plant_ID
    - name: plant_name
    - fuel_type: fuel type (e.g., "Coal", "Natural Gas")
    - capacity: capacity in MW
    - capacity_factor: capacity factor for 2025
    
    Args:
        generators_df: DataFrame from load_active_generators()
    
    Returns:
        List of opponent agent dicts
    """
    opponents = []
    for _, row in generators_df.iterrows():
        opponent = {
            "id": row["Power_Plant_ID"],
            "name": row["plant_name"],
            "fuel_type": row["fuel_type"],
            "capacity": float(row["capacity"]),
            "capacity_factor": float(row["capacity_factor_2025"]),
        }
        opponents.append(opponent)
    return opponents

def allocate_load_to_opponents(
    total_load: float,
    fuel_type: str,
    opposed_of_fuel_type: list[dict],
) -> dict[str, float]:
    """
    Allocate a fuel type's share of the total load among opponents of that fuel type,
    proportional to their capacity.
    
    Args:
        total_load: Total system load in MW
        fuel_type: The fuel type (e.g., "Coal", "Natural Gas")
        opponents_of_fuel_type: List of opponent dicts filtered to this fuel_type
    
    Returns:
        Dict mapping opponent ID to allocated load in MW
    """
    allocation = {}
    
    # If no opponents of this fuel type, return empty dict
    if not opposed_of_fuel_type:
        return allocation
    
    # Calculate total capacity of opponents of this fuel type
    total_capacity = sum(opp["capacity"] for opp in opposed_of_fuel_type)
    
    if total_capacity <= 0:
        # Distribute equally if no capacity info
        per_opponent = total_load / len(opposed_of_fuel_type)
        for opp in opposed_of_fuel_type:
            allocation[opp["id"]] = per_opponent
    else:
        # Allocate proportional to capacity
        for opp in opposed_of_fuel_type:
            share = opp["capacity"] / total_capacity
            allocated = total_load * share
            allocation[opp["id"]] = allocated
    
    return allocation

def segment_load_by_fuel_and_capacity(
    load_df: pd.DataFrame,
    fuel_mix_df: pd.DataFrame,
    generators_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Segment historical load by fuel type (using fuel_mix) and further by generator capacity.
    
    For each hourly timestamp:
    1. Use fuel_mix_df to allocate the total load to each fuel type
    2. For each fuel type, allocate its share to individual generators based on capacity
    
    Args:
        load_df: Historical load data with 'period' datetime column and HIST_LOAD_COL (total load)
        fuel_mix_df: Fuel mix data with 'datetime' and fuel type columns
        generators_df: Active generators data with fuel_type and capacity
    
    Returns:
        DataFrame with columns:
        - datetime: UTC timestamp
        - opponent_id: Generator ID
        - opponent_name: Generator name
        - fuel_type: Generator fuel type
        - allocated_load_mw: Load allocated to this generator
    """
    # Build opponent agents
    opponents = build_opponent_agents(generators_df)
    
    # Group opponents by fuel type
    opponents_by_fuel = {}
    for opp in opponents:
        fuel = opp["fuel_type"]
        if fuel not in opponents_by_fuel:
            opponents_by_fuel[fuel] = []
        opponents_by_fuel[fuel].append(opp)
    
    # Merge load and fuel mix on datetime
    load_df_copy = load_df.copy()
    if "period" in load_df_copy.columns:
        load_df_copy["datetime"] = load_df_copy["period"]
    load_df_copy["datetime"] = pd.to_datetime(load_df_copy["datetime"], utc=True)
    
    fuel_mix_copy = fuel_mix_df.copy()
    fuel_mix_copy["datetime"] = pd.to_datetime(fuel_mix_copy["datetime"], utc=True)
    
    merged = load_df_copy.merge(fuel_mix_copy, on="datetime", how="inner")
    
    # Segment each row
    rows = []
    fuel_type_cols = [col for col in fuel_mix_df.columns if col not in ["datetime", "Total_MW"]]
    
    for _, row in merged.iterrows():
        dt = row["datetime"]
        total_load = float(row.get(HIST_LOAD_COL, row.get("Total_MW", 0)))
        
        # For each fuel type, allocate to generators
        for fuel_type in fuel_type_cols:
            fuel_proportion = float(row.get(fuel_type, 0))
            fuel_load = total_load * fuel_proportion
            
            # Get opponents of this fuel type
            opps = opponents_by_fuel.get(fuel_type, [])
            
            # Allocate this fuel type's load among its generators
            allocation = allocate_load_to_opponents(fuel_load, fuel_type, opps)
            
            for opponent_id, allocated_mw in allocation.items():
                # Find opponent details
                opp = next((o for o in opps if o["id"] == opponent_id), None)
                if opp:
                    rows.append({
                        "datetime": dt,
                        "opponent_id": opponent_id,
                        "opponent_name": opp["name"],
                        "fuel_type": fuel_type,
                        "capacity_mw": opp["capacity"],
                        "allocated_load_mw": allocated_mw,
                    })
    
    segmented_df = pd.DataFrame(rows)
    return segmented_df

# ============================================================================
# SUPPLY STACK AND OPPONENT BIDDING
# ============================================================================
# Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
SUPPLY_STACK_ORDER = ["Wind", "Solar", "Hydro", "Nuclear", "Biomass", "Coal", "Natural Gas", "Other"]

def generate_opponent_bids_for_timestep(
    timestamp: pd.Timestamp,
    segmented_load_df: pd.DataFrame,
    lmp_df: pd.DataFrame,
    fuel_type_order: list[str] = None,
    price_noise_std: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Generate opponent bids for a single timestep using the supply stack model.
    
    For each fuel type in the supply stack order:
    1. Calculate the cumulative proportion of load up to that fuel type
    2. Determine the price range: [LMP * cumulative_prop, LMP * (cumulative_prop + fuel_prop)]
    3. For each opponent of that fuel type, sample a price from Gaussian in this range
    4. Create a bid with quantity = allocated_load_mw and sampled price
    
    Args:
        timestamp: UTC datetime for this timestep
        segmented_load_df: Full segmented load data (output of segment_load_by_fuel_and_capacity)
        lmp_df: Price data with datetime and PRICE_COL columns
        fuel_type_order: Ordered list of fuel types (top to bottom in stack).
                         Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
        price_noise_std: Standard deviation for Gaussian price sampling
        rng: Random number generator
        
    Returns:
        List of opponent bid dicts with keys:
        - opponent_id, opponent_name, fuel_type, bid_price, bid_quantity, allocated_load_mw
    """
    if fuel_type_order is None:
        # Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
        fuel_type_order = SUPPLY_STACK_ORDER
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Get LMP for this timestamp
    ts_lmp_rows = lmp_df[lmp_df["datetime"] == timestamp]
    if ts_lmp_rows.empty:
        return []
    lmp = float(ts_lmp_rows[PRICE_COL].iloc[0])
    
    # Filter segmented load for this timestamp
    ts_loads = segmented_load_df[segmented_load_df["datetime"] == timestamp].copy()
    if ts_loads.empty:
        return []
    
    total_load = ts_loads["allocated_load_mw"].sum()
    if total_load <= 0:
        return []
    
    bids = []
    cumulative_prop = 0.0
    
    for fuel_type in fuel_type_order:
        fuel_data = ts_loads[ts_loads["fuel_type"] == fuel_type]
        if fuel_data.empty:
            continue
        
        fuel_total_load = fuel_data["allocated_load_mw"].sum()
        fuel_prop = fuel_total_load / total_load
        
        # Price range for this fuel type in the supply stack
        # Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
        price_low = lmp * cumulative_prop
        price_high = lmp * (cumulative_prop + fuel_prop)
        price_mid = (price_low + price_high) / 2
        
        # Generate bid for each opponent in this fuel type
        for _, opp_row in fuel_data.iterrows():
            # Sample price from Gaussian centered at midpoint of fuel type's price range
            bid_price = float(rng.normal(price_mid, price_noise_std))
            bid_price = float(np.clip(bid_price, price_low, price_high))
            
            bids.append({
                "opponent_id": opp_row["opponent_id"],
                "opponent_name": opp_row["opponent_name"],
                "fuel_type": fuel_type,
                "capacity_mw": float(opp_row["capacity_mw"]),
                "allocated_load_mw": float(opp_row["allocated_load_mw"]),
                "bid_price": bid_price,
                "bid_quantity": float(opp_row["allocated_load_mw"]),
            })
        
        cumulative_prop += fuel_prop
    
    return bids

def extract_rl_bids_from_actions(
    action_indices: list[int],
    action_space: ActionSpace,
) -> list[dict]:
    """
    Extract price and quantity bids from RL agent action indices.
    
    Current RL agents: Natural Gas only.
    Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
    
    Args:
        action_indices: List of discrete action indices (one per RL agent)
        action_space: ActionSpace object to decode indices
        
    Returns:
        List of dicts with keys: agent_idx, bid_price, bid_quantity
    """
    rl_bids = []
    for agent_idx, action_idx in enumerate(action_indices):
        price, quantity = action_space.decode_to_values(int(action_idx))
        rl_bids.append({
            "agent_idx": agent_idx,
            "bid_price": float(price),
            "bid_quantity": float(quantity),
        })
    return rl_bids

def clear_market_with_stack_competition(
    rl_bids: list[dict],
    opponent_bids: list[dict],
    total_historical_load: float,
    clearing_price: float | None = None,
) -> dict:
    """
    Clear the market using a merit-order auction (lowest price first).
    
    Bids are sorted by price ascending (merit order). Bids are filled from lowest price up
    until total demand (historical load) is met. All cleared bids pay the clearing price
    (uniform price auction), which is the price of the highest-accepted bid (marginal price).
    
    The total cleared quantity always equals total_historical_load.
    
    Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
    
    Args:
        rl_bids: List of RL agent bids with keys: agent_idx, bid_price, bid_quantity
        opponent_bids: List of opponent bids with keys: opponent_id, bid_price, bid_quantity, ...
        total_historical_load: Total system demand (sum of all historical load)
        clearing_price: If provided, fix the clearing price (otherwise use auction price)
        
    Returns:
        Dict with keys:
        - clearing_price: price all winners pay (marginal price)
        - rl_cleared: list of cleared quantities per RL agent [q0, q1, ...]
        - opponent_cleared: dict mapping opponent_id -> cleared quantity
        - marginal_price: the price of the highest accepted bid (clearing price)
        - total_cleared: sum of all cleared quantities
    """
    # Combine all bids with type tags
    all_bids = []
    for bid in rl_bids:
        all_bids.append({
            "type": "rl",
            "agent_idx": bid["agent_idx"],
            "price": float(bid["bid_price"]),
            "quantity": float(bid["bid_quantity"]),
        })
    
    for bid in opponent_bids:
        all_bids.append({
            "type": "opponent",
            "opponent_id": bid["opponent_id"],
            "price": float(bid["bid_price"]),
            "quantity": float(bid["bid_quantity"]),
        })
    
    # Sort by price ascending (merit order: lowest cost generators first)
    # Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
    all_bids.sort(key=lambda x: x["price"], reverse=False)
    
    # Fill bids from lowest price up until demand is satisfied
    total_cleared = 0.0
    cleared_bids = []
    marginal_price = 0.0
    
    for bid in all_bids:
        if total_cleared >= total_historical_load:
            break
        
        remaining_demand = total_historical_load - total_cleared
        quantity_to_clear = min(bid["quantity"], remaining_demand)
        
        if quantity_to_clear > 1e-6:  # Only if quantity is non-negligible
            marginal_price = bid["price"]  # Price of highest-accepted (last-filled) bid
            cleared_bids.append({
                **bid,
                "cleared_quantity": quantity_to_clear,
            })
            total_cleared += quantity_to_clear
    
    # Uniform price auction: all winners pay the marginal (clearing) price
    if clearing_price is None:
        clearing_price = marginal_price if marginal_price > 0 else 0.0
    
    # Extract results by agent
    rl_cleared = [0.0] * len(rl_bids)
    opponent_cleared = {bid["opponent_id"]: 0.0 for bid in opponent_bids}
    
    for cleared_bid in cleared_bids:
        if cleared_bid["type"] == "rl":
            idx = cleared_bid["agent_idx"]
            rl_cleared[idx] = cleared_bid["cleared_quantity"]
        else:
            opponent_cleared[cleared_bid["opponent_id"]] = cleared_bid["cleared_quantity"]
    
    return {
        "clearing_price": clearing_price,
        "rl_cleared": rl_cleared,
        "opponent_cleared": opponent_cleared,
        "marginal_price": marginal_price,
        "total_cleared": total_cleared,
    }

def infer_market_params_from_lmp(
    lmp_df: pd.DataFrame,
    price_col: str,
    *, # Keyword-only arguments
    mc_tail_q: float = 0.25, # use lower 25% of lmp (i.e. when market is slack)
    vol_clip_q: float = 0.995, # clip extreme prices
) -> tuple[float, float]:
    '''Infer marginal cost from historical LMP data'''
    price_series = pd.to_numeric(lmp_df[price_col], errors="coerce").dropna()

    cutoff_price = price_series.quantile(mc_tail_q)
    marginal_cost = price_series[price_series <= cutoff_price].mean()

    diffs = price_series.diff().dropna()
    cap = diffs.abs().quantile(vol_clip_q)
    diffs = diffs.clip(-cap, cap)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    price_noise_std = mad * 1.4826  # converts MAD to std under normality assumption

    if not np.isfinite(marginal_cost):
        marginal_cost = float(price_series.quantile(0.10))
    if not np.isfinite(price_noise_std):
        price_noise_std = float(diffs.std(ddof=0)) if len(diffs) else 0.0

    return marginal_cost, price_noise_std

def build_world(load_df: pd.DataFrame, lmp_df: pd.DataFrame) -> tuple[State, ActionSpace, MarketModel]:
    state, _ = build_state_and_discretizers(load_df, lmp_df)
    action_space = make_action_space(lmp_df)

    price_edges = action_space.price_disc.edges_
    if price_edges is None:
        raise RuntimeError("price_disc.edges_ is None; call fit(...) first.")
    
    marginal_cost, price_noise_std = infer_market_params_from_lmp(lmp_df, PRICE_COL)
    market_params = MarketParams(
        marginal_cost=marginal_cost, 
        price_noise_std=price_noise_std,
        min_price=float(price_edges[0]),
        max_price=float(price_edges[-1]),
    )
    market_model = MarketModel(action_space, market_params)
    return state, action_space, market_model

def build_world_and_data():
    load_df, lmp_df = load_data()
    state, action_space, market_model = build_world(load_df, lmp_df)
    
    apply_demand_scale_to_state(state, float(args.demand_scale))
    state.apply()

    return state, action_space, market_model, load_df, lmp_df

from typing import Any, List, Tuple, TypeAlias

def build_agents(n_agents: int, num_actions: int, *, seed: int | None = None) -> List[TabularQLearningAgent]:
    agents: List[TabularQLearningAgent] = []
    for i in range(n_agents):
        a = TabularQLearningAgent(num_actions=num_actions)
        # If your agent has a seed() method, give each agent a different seed for tie-breaking diversity
        if seed is not None and hasattr(a, "seed"):
            a.seed(int(seed) + 1000 * i)
        agents.append(a)
    return agents

def select_and_project_actions(
    agents: List[TabularQLearningAgent],
    obs: Observation,
    action_space: ActionSpace,
    *,
    max_quantity: float,
    max_notional: float,
    frozen: bool = False,
    agent_policies: List[dict] | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[List[int], List[dict]]:
    """
    "Simultaneous" action submission: each agent acts from the same obs,
    we project each action to feasibility, then return the final action indices.
    """

    action_indices: List[int] = []
    clip_infos: List[dict] = []

    for ag_idx, agent in enumerate(agents):
        raw_idx: int
        # If policies are frozen and a per-agent policy mapping exists, prefer it
        if frozen and agent_policies is not None:
            # extract state_key from obs
            state_key = obs.get("state_key")
            if state_key is not None:
                # If the agent has a deterministic action for this state, use it
                pol = agent_policies[ag_idx]
                if pol is not None and state_key in pol:
                    raw_idx = int(pol[state_key])
                else:
                    raw_idx = int(agent.select_softmax_action(state_key, temperature=obs.get("temperature")))
            else:
                raw_idx = agent.act(obs)
        else:
            raw_idx = agent.act(obs)
        aidx, clip_info = action_space.project_to_feasible(
            raw_idx,
            max_quantity=max_quantity,
            max_notional=max_notional,
        )
        action_indices.append(int(aidx))
        clip_infos.append(clip_info)

    return action_indices, clip_infos


def train(n_episodes = 20, *, seed: int | None = None, overrides: dict | None = None) -> tuple[List[TabularQLearningAgent], State, ActionSpace, MarketModel, list[dict]]:
    # For ablation (temporary change for single run)
    if overrides is not None:
        for k, v in overrides.items():
            setattr(args, k, v)

    load_df, lmp_df = load_data()
    forecast_series = _load_ercot_forecast_series()
    state, action_space, market_model = build_world(load_df, lmp_df)
    print(f"[Market Model] Inferred marginal cost: {market_model.params.marginal_cost:.4f} | price noise std: {market_model.params.price_noise_std:.4f}")
    apply_demand_scale_to_state(state, float(args.demand_scale))
    state.apply()

    # Load fuel mix and generators for supply stack competition.
    # Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
    fuel_mix_df = load_fuel_mix()
    generators_df = load_active_generators()
    segmented_load_df = segment_load_by_fuel_and_capacity(load_df, fuel_mix_df, generators_df)
    
    if args.verbose:
        print(f"[Stack] Loaded {len(generators_df)} generators across {len(fuel_mix_df)} timesteps")
        print(f"[Stack] Segmented load has {len(segmented_load_df)} opponent bid rows")

    if seed is not None:
        np.random.seed(seed)
    
    # RNG for opponent bid generation and other randomness
    rng = np.random.default_rng(seed)

    price_q = float(lmp_df[PRICE_COL].abs().quantile(args.max_notional_q))
    max_notional = float(MAX_BID_QUANTITY_MW * price_q)

    n_agents = int(getattr(args, "n_agents", 1))
    agents = build_agents(n_agents, num_actions=action_space.n_actions, seed=seed)

    # Per-agent deterministic policies used when policies are frozen.
    # Each is a mapping: StateKey -> action_index
    agent_policies: List[dict] = [{} for _ in agents]

    if(args.verbose):
        print(f"[Risk] max_notional_q={args.max_notional_q} | price_q={price_q:.4f} | max_notional={max_notional:.4f}")

    ##agent = TabularQLearningAgent(num_actions=action_space.n_actions)

    ##if seed is not None and hasattr(agent, "seed"):
      ##  agent.seed(seed)

    marginal_cost = float(market_model.params.marginal_cost)
    cost_plus_policy = CostPlusMarkupPolicy(
        action_space=action_space,
        marginal_cost=marginal_cost,
        markup=float(args.markup),
        quantity_mw=MAX_BID_QUANTITY_MW,
    )

    episode_logs: list[dict] = []

    # Initialize metrics tracker module for multi-agent environment.
    metrics = MetricsTracker(n_agents=n_agents, agent_names = [f"RL_Agent_{i}" for i in range(n_agents)], action_space=action_space)

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been intialized")
    
    episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))
    episode_starts = episode_starts[:n_episodes] # Limit to n_episodes

    for ep_idx, start_idx in enumerate(episode_starts):
        if args.verbose:
            print(f"\n=== EPISODE {ep_idx+1}/{len(episode_starts)} | start_idx={start_idx} ===")

        # If policy-freeze is enabled, periodically recompute greedy policies
        # from the current Q-tables and apply inertia to avoid thrashing.
        if bool(args.policy_freeze_enabled) and int(args.policy_freeze_k) > 0:
            k = int(args.policy_freeze_k)
            # Update policies at episode indices that are multiples of k
            if (ep_idx % k) == 0:
                if args.verbose:
                    print(f"[Policy Update] ep={ep_idx} | recomputing greedy policies with inertia={args.policy_inertia_keep}")
                for i, ag in enumerate(agents):
                    # Greedy policy derived from current Q (almost-zero temperature)
                    new_pol: dict = ag.extract_policy(temperature=1e-6)
                    old_pol: dict = agent_policies[i] if agent_policies[i] is not None else {}
                    merged: dict = {}
                    # union of states
                    states = set(old_pol.keys()) | set(new_pol.keys())
                    for s in states:
                        old_a = old_pol.get(s)
                        new_a = new_pol.get(s)
                        if old_a is None:
                            merged[s] = new_a
                        elif new_a is None:
                            merged[s] = old_a
                        elif old_a == new_a:
                            merged[s] = new_a
                        else:
                            keep_prob = float(args.policy_inertia_keep)
                            if rng.random() < keep_prob:
                                merged[s] = old_a
                            else:
                                merged[s] = new_a
                    agent_policies[i] = merged
        cumulative_reward = 0

        state.episode_start = start_idx
        episode_start_ts = state.raw_state_data.index[start_idx]

        h_end = episode_start_ts + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
        f_slice = forecast_series.loc[(forecast_series.index >= episode_start_ts) & (forecast_series.index <= h_end)]
        forecast_df = f_slice.rename(FORECAST_LOAD_COL).reset_index()
        inject_epsisode_forecast(state, forecast_df)

        # Re-discretize now that we have forecast data
        state.apply()
        obs = state.reset(new_episode=False)
        state_key = agents[0].state_to_key(obs)

        done = False
        step_counter = 0

        while not done:
            temp = get_episode_temperature(ep_idx)
            ts = state.timestamps[state.ptr]
            step_obs: Observation = {
                "timestamp": ts,
                "state_key": state_key,
                "temperature": temp,
            }

            action_indices, clip_infos = select_and_project_actions(
                agents,
                step_obs,
                action_space,
                max_quantity=MAX_BID_QUANTITY_MW,
                max_notional=max_notional,
                frozen=bool(args.policy_freeze_enabled),
                agent_policies=agent_policies,
                rng=rng,
            )

            # Bidding 24 hours in advance
            delivery_time = ts + pd.Timedelta(hours=24)

            if delivery_time not in state.raw_state_data.index:
                # If we don't have data for the delivery time, end episode
                print(f"  No data for delivery time {delivery_time}, ending episode.")
                done = True
                break

            delivery_row = state.raw_state_data.loc[delivery_time]

            price_val = delivery_row.get(PRICE_COL, np.nan)
            if pd.isna(price_val):
                price_val = float(lmp_df[PRICE_COL].iloc[-1]) # Fallback to last known price
        
            # Historical load for residual clearing
            historical_load_mw = delivery_row.get(HIST_LOAD_COL, np.nan)
            historical_load_mw = float(historical_load_mw) if pd.notna(historical_load_mw) else 0.0

            # Generate opponent bids for this delivery time
            # Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
            opponent_bids = generate_opponent_bids_for_timestep(
                delivery_time,
                segmented_load_df,
                lmp_df,
                fuel_type_order=SUPPLY_STACK_ORDER,
                price_noise_std=float(args.supply_stack_price_noise_std),
                rng=rng,
            )
            
            # Extract RL agent bids from action indices
            rl_bids = extract_rl_bids_from_actions(action_indices, action_space)
            
            # Clear market with both RL and opponent bids using supply stack model
            # Supply stack order (top to bottom): Wind / Solar / Hydro / Nuclear / Biomass / Coal / Natural Gas / Other
            clearing_result = clear_market_with_stack_competition(
                rl_bids,
                opponent_bids,
                total_historical_load=historical_load_mw,
                clearing_price=None,  # Let auction determine clearing price
            )
            
            clearing_price = clearing_result["clearing_price"]
            rl_cleared = clearing_result["rl_cleared"]

            # Warm start Q-values if needed
            if args.warm_start_q:
                baseline_action_idx = int(cost_plus_policy.act(step_obs))
                for ag in agents:
                    if state_key not in ag.Q:
                        baseline_reward = market_model.peek_reward_from_action(baseline_action_idx, float(price_val))
                        margin = 1.0
                        ag.warm_start_state(
                            state_key,
                            preferred_action=baseline_action_idx,
                            preferred_q=baseline_reward + margin,
                            other_q=baseline_reward - margin,
                            only_if_unseen=True,
                        )

            # Calculate rewards based on cleared quantities
            marginal_cost = float(market_model.params.marginal_cost)
            rewards = [
                (clearing_price - marginal_cost) * rl_cleared[i]
                for i in range(len(agents))
            ]
            
            # Compute residual share (rho) for metrics
            total_rl_cleared = sum(rl_cleared)
            rho_for_metrics = total_rl_cleared / historical_load_mw if historical_load_mw > 0 else 0.0

            # Log Metrics
            metrics.log_step(
                episode=ep_idx, step=step_counter, timestamp=ts,
                action_indices=action_indices, rewards=rewards,
                clearing_price=clearing_price, demand_mw=historical_load_mw,
                rho=rho_for_metrics, clip_infos=clip_infos,
            )

            if args.risk_penalty_lambda > 0.0:
                for i, clip_info in enumerate(clip_infos):
                    if bool(clip_info.get("clipped", False)):                        
                        orig_notional = float(clip_info["original_notional"])
                        if max_notional > 1e-12:
                            severity = max(0.0, (orig_notional - max_notional) / max_notional)
                            if args.verbose: 
                                print(f"[Risk Penalty] step={step_counter} | orig_notional={orig_notional:.4f} | max_notional={max_notional:.4f} | severity={severity:.4f}")
                        else:
                            severity = 0.0
                        penalty = float(args.risk_penalty_lambda * (1.0 + severity))
                        rewards[i] -= penalty

            next_obs, _, done, info = state.step()
            next_state_key = agents[0].state_to_key(next_obs)
            next_step_obs: Observation = {
                "timestamp": state.timestamps[state.ptr],
                "state_key": next_state_key,
            }

            for i, ag in enumerate(agents):
                ag.update(
                    step_obs,
                    int(action_indices[i]),
                    float(rewards[i]),
                    next_step_obs,
                    done,
                )

            cumulative_reward += float(np.sum(rewards))

            if args.verbose:
                print(
                    f"[EP {ep_idx+1} | Step {step_counter}] ts={ts} | "
                    f"state_key={state_key} | "
                    f"P={clearing_price:.2f} | demand={historical_load_mw:.2f} | rho={rho_for_metrics:.3f} | "
                    f"actions={action_indices} | rewards={np.round(rewards, 4).tolist()} | "
                    f"cum_total_reward={cumulative_reward:.4f}"
                )

            state_key = next_state_key
            obs = next_obs
            step_counter += 1

            if step_counter >= state.window_size:
                done = True
        
        # Close Episode
        metrics.close_episode(ep_idx)

        if args.verbose:
            print(f"Episode {ep_idx+1} finished after {step_counter} steps")
        
        # lightweight progress (works even when verbose is off)
        if args.progress_every and ((ep_idx + 1) % args.progress_every == 0):
            seed_str = seed if seed is not None else "NA"
            print(f"\n=== SEED {seed_str} | EPISODE {ep_idx+1}/{len(episode_starts)} | start_idx={start_idx} ===")
        
        episode_logs.append({
            "seed": seed if seed is not None else -1,
            "warm_start_q": bool(args.warm_start_q),
            "episode": ep_idx,
            "cumulative_reward": float(cumulative_reward),
        })

    # Export multi-agent metrics if requested
    if args.export_metrics:
        _ = export_multi_agent_metrics(metrics)
    
    with open("q_table.pkl", "wb") as f:
        pickle.dump([ag.Q for ag in agents], f)

    with open("policy.pkl", "wb") as f:
        pickle.dump([ag.extract_policy() for ag in agents], f)
    
    print("Training complete. Q-table and policy saved.")
    return agents, state, action_space, market_model, episode_logs

def run_baselines(
        *, # Keyward-only arguments since complex signature
        baseline: str,
        n_episodes: int,
        markup: float,
        quantile:float,
):
    load_df, lmp_df = load_data()
    state, action_space, market_model = build_world(load_df, lmp_df)  
    
    marginal_cost, _ = infer_market_params_from_lmp(lmp_df, PRICE_COL)

    policy: Policy
    if baseline == "cost_plus":
        policy = CostPlusMarkupPolicy(
            action_space = action_space,
            marginal_cost = marginal_cost,
            markup = markup,
            quantity_mw = MAX_BID_QUANTITY_MW
        )
    elif baseline == "hist_quantile":
        policy = HistoricalQuantilePolicy(
            action_space = action_space,
            lmp_df= lmp_df,
            quantile = quantile,
            quantity_mw = MAX_BID_QUANTITY_MW
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    df = run_policy_on_episodes(
        policy=policy,
        state=state,
        market_model=market_model,
        load_df=load_df,
        lmp_df=lmp_df,
        n_episodes=n_episodes,
        inject_forecast_fn=inject_epsisode_forecast,
        verbose = args.verbose
    )

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--mode", choices=["train", "baseline"], default="train")

    p.add_argument("--n_episodes", type=int, default=20)

    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # baseline specific
    p.add_argument("--baseline", choices=["cost_plus", "hist_quantile"], default="cost_plus")
    p.add_argument("--markup", type=float, default=10.0, help="Markup for cost_plus baseline")
    p.add_argument("--quantile", type=float, default=0.7, help="Quantile for hist_quantile baseline")

    # risk constraints
    p.add_argument("--max_notional_q", type=float, default=0.95, help="Quantile of |LMP| used to set max_notional")
    p.add_argument("--risk_penalty_lambda", type=float, default=0.0, help="Penalty lambda for risk constraint violations")
    p.add_argument("--max_drawdown", type=float, default=float("inf"), help="Maximum allowable drawdown over an episode. inf disables.")

    p.add_argument(
        "--warm_start_q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm start unseen-state Q values using the cost+markup baseline (default: enabled).",
    )
    
    p.add_argument(
        "--run_all_ablations",
        action="store_true",
        help="Run all ablation studies (warm start, risk constraint, temperature) and save CSV/plots."
    )
    p.add_argument("--ablation_seeds", type=int, default=5)
    p.add_argument("--ablation_episodes", type=int, default=50)
    p.add_argument("--ablation_out_csv", type=str, default="warm_start_ablation.csv")
    p.add_argument("--ablation_out_png", type=str, default="warm_start_ablation.png")
    p.add_argument("--temperature_mode", choices=["fixed", "qgap", "exp_decay"], default="fixed")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_min", type=float, default=0.1)
    p.add_argument("--temperature_decay", type=float, default=0.995)
    p.add_argument("--risk_lambda_on", type=float, default=1.0)
    
    # Policy-freeze / inertia options (disabled by default)
    p.add_argument("--policy_freeze_enabled", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable periodic policy freeze: hold policies fixed for K episodes while Q updates. (default: enabled)")
    p.add_argument("--policy_freeze_k", type=int, default=5,
                   help="Number of episodes to hold policies fixed between greedy policy updates.")
    p.add_argument("--policy_inertia_keep", type=float, default=0.9,
                   help="When updating policy, probability of keeping previous action for a state (inertia).")

    p.add_argument(
    "--demand_scale",
    type=float,
    default=1.0,
    help="Multiply HIST_LOAD_COL and FORECAST_LOAD_COL by this factor before discretization."
)

    p.add_argument(
        "--plot_demand_perturbation",
        action="store_true",
        help="Evaluate a saved policy under demand scales and save performance-vs-scale plot."
    )

    p.add_argument("--demand_scales", type=str, default="0.9,1.0,1.1",
                help="Comma-separated demand scales for perturbation sweep.")
    p.add_argument("--eval_policy_path", type=str, default="policy.pkl",
                help="Path to saved deterministic policy mapping.")
    p.add_argument("--eval_q_table_path", type=str, default="q_table.pkl",
                help="Optional: Q-table fallback if policy missing a state.")
    
    p.add_argument("--run_master_ablations", action="store_true",
               help="Run warm-start + risk + temperature ablations AND the demand perturbation sweep.")
    
    p.add_argument("--progress_every", type=int, default=25,
               help="Print a progress line every N steps (0 disables).")
    
    p.add_argument("--n_agents", type=int, default=2, help="Number of bidding agents.")
    
    p.add_argument("--supply_stack_price_noise_std", type=float, default=2.0,
               help="Standard deviation for Gaussian price sampling in supply stack model.")

    p.add_argument("--rho_min", type=float, default=0.1, help="rho_min: residual share when price is high")
    p.add_argument("--rho_max", type=float, default=0.9, help="rho_max: residual share when price is low")
    p.add_argument("--rho_k", type=float, default=0.05, help="k: steepness of rho(P)")
    p.add_argument("--rho_p0", type=float, default=50.0, help="p0: switch price of rho(P)")

    p.add_argument("--export_metrics", action="store_true", default=False,
               help="Export metrics and plots to Analysis/Metrics/ directory (default: disabled)")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.run_all_ablations or args.run_master_ablations:
        cfg = AllAblationsConfig(
            seeds=args.ablation_seeds,
            episodes=args.ablation_episodes,
            risk_lambda_on=args.risk_penalty_lambda
        )
        run_all_ablations(train_fn=train, args=args, cfg=cfg)

    if args.plot_demand_perturbation or args.run_master_ablations:
        scales = [float(x.strip()) for x in args.demand_scales.split(",")]
        cfg = DemandPerturbationConfig(
            scales=scales,
            seeds=args.ablation_seeds,
            episodes=args.ablation_episodes,
        )
        run_demand_perturbation_sweep(
            build_world_and_data_fn=build_world_and_data,
            inject_forecast_fn=inject_epsisode_forecast,
            args=args,
            cfg=cfg,
            policy_path=args.eval_policy_path,
            q_table_path=args.eval_q_table_path
        )
    
    if args.mode == "train":
        train(n_episodes=args.n_episodes)
    
    if args.mode == "baseline":
        run_baselines(
            baseline=args.baseline,
            n_episodes=args.n_episodes,
            markup=args.markup,
            quantile=args.quantile
        )