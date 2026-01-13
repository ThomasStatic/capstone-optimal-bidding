from typing import Callable
import numpy as np
import pandas as pd
from shell.evaluations.policy_types import Policy
from shell.linear_approximator import PRICE_COL
from shell.load_sarimax_projections import SARIMAXLoadProjections


def run_policy_on_episodes(
        *, # Keyword-only arguments since complex signature
        policy: Policy,
        state,
        market_model,
        load_df: pd.DataFrame,
        lmp_df: pd.DataFrame,
        n_episodes: int,
        inject_forecast_fn: Callable, # injecting function from main.py
        verbose: bool = False
) -> pd.DataFrame:
    """
    Mirrors main.train() episode structure but uses policy.act(...) instead of RL.

    Requirements:
      - policy.act(obs) -> action_idx (int)
      - obs includes {"timestamp": ts} (UTC)
      - market_model.clear_market_from_action(action_idx, forecast_price) -> (..., ..., reward)
    """
    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been initialized")
    
    episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))
    episode_starts = episode_starts[:n_episodes]

    rows = []

    for ep_idx, start_idx in enumerate(episode_starts):
        state.episode_start = start_idx
        episode_start_ts = state.raw_state_data.index[start_idx]
        
        if verbose:
            print(f"\n=== Episode {ep_idx} ===")
            print(f"Episode start index: {start_idx}")
            print(f"Episode start timestamp: {episode_start_ts}")
            print(f"Policy: {policy.__class__.__name__}")

        # Build SARIMAX history only using data prior to episode start
        history_mask = load_df["period"] < episode_start_ts
        history_for_model = load_df.loc[history_mask].copy()
        if history_for_model.empty:
            history_for_model = load_df.iloc[:state.window_size].copy()
        sarimax = SARIMAXLoadProjections(history_for_model)
        forecast_df = sarimax.get_forecast_df()

        if verbose:
            print(f"Forecast rows injected: {len(forecast_df)}")
            print(f"Forecast head:\n{forecast_df.head(2)}")

        # Inject forecast into state, then discretize
        inject_forecast_fn(state, forecast_df)
        state.apply()
        _ = state.reset(new_episode=False)

        done = False
        step_counter = 0
        ep_profit = 0.0
        step_profits: list[float] = []

        MAX_DEBUG_STEPS = 3

        while not done:
            ts = state.timestamps[state.ptr]
            delivery_time = ts + pd.Timedelta(hours=24)

            if delivery_time not in state.raw_state_data.index:
                done = True
                break

            delivery_row = state.raw_state_data.loc[delivery_time]
            forecast_price = delivery_row.get(PRICE_COL, np.nan)
            if pd.isna(forecast_price):
                # fallback: last known price in lmp_df
                forecast_price = float(lmp_df[PRICE_COL].iloc[-1])
            
            if verbose and step_counter < MAX_DEBUG_STEPS:
                print(
                    f"[t={ts}] "
                    f"delivery={delivery_time} | "
                    f"forecast_price={forecast_price:.2f}"
                )

            action_idx = policy.act({"timestamp": ts})
            _, _, reward = market_model.clear_market_from_action(action_idx, float(forecast_price))

            if verbose and step_counter < MAX_DEBUG_STEPS:
                bid_price, bid_qty = market_model.action_space.decode_to_values(action_idx)
                print(
                    f"  action_idx={action_idx} | "
                    f"bid_price={bid_price:.2f}, bid_qty={bid_qty:.2f}"
                )
                print(f"  reward/profit={reward:.2f}")

            profit = float(reward)
            ep_profit += profit
            step_profits.append(profit)

            _, _, done, _info = state.step()
            step_counter += 1
            if step_counter >= state.window_size:
                done = True

            # downside proxy: per-step CVaR(10%)
            sp = np.array(step_profits, dtype=float) if step_profits else np.array([0.0])
            k = max(1, int(0.10 * len(sp)))
            cvar10 = float(np.mean(np.sort(sp)[:k]))

            rows.append(
            {
                "episode": ep_idx,
                "profit": ep_profit,
                "cvar10_step": cvar10,
                "n_steps": len(step_profits),
            }
            )

            if verbose:
                print(
                    f"Episode {ep_idx} finished | "
                    f"profit={ep_profit:.2f} | "
                    f"steps={len(step_profits)} | "
                    f"CVaR10={cvar10:.2f}"
                )
            

    return pd.DataFrame(rows)