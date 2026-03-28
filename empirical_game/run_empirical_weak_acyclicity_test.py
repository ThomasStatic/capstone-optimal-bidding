from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence, cast

import numpy as np
import pandas as pd

from empirical_game.best_response import (
    BestResponseConfig,
    DecisionRule,
    best_reply_edges,
    best_response_gaps,
    deviation_table,
    sink_profiles,
)
from empirical_game.graph_analysis import GraphAnalysisResult, analyze_weak_acyclicity
from empirical_game.library_builder import LibraryBuildOptions, build_policy_libraries
from empirical_game.payoff_estimator import MonteCarloConfig, MonteCarloPayoffEstimator
from empirical_game.policy_wrappers import (
    PolicyWrapper,
    make_policy_wrapper_from_callable,
    make_policy_wrapper_from_map,
)
from empirical_game.profile_space import JointProfile, ProfileSpace, build_profile_space
from empirical_game.reports import export_tables, local_robustness_around_profile, sink_quality_summary
from shell.agent_interface import Observation, StateKey, load_saved_agents, state_vec_to_key
from shell.baselines.cost_plus_markup import CostPlusMarkupPolicy
from shell.baselines.historical_quantile import HistoricalQuantilePolicy


@dataclass(frozen=True)
class CapstoneSimulatorConfig:
    n_agents: int
    n_eval_episodes: int
    load_cache_path: str = "load_cache.csv"
    demand_scale: float = 1.0
    lmp_competition_elasticity: float = 0.15
    supply_stack_price_noise_std: float = 2.0
    max_notional_q: float = 0.95


class CapstoneReducedGameSimulator:
    """Adapter that reuses `shell.main` logic for reduced-game profile evaluation."""

    def __init__(self, cfg: CapstoneSimulatorConfig) -> None:
        import shell.main as sim_main

        self.cfg = cfg
        self.sim_main = sim_main
        self.n_agents = int(cfg.n_agents)

        self._set_main_args()

        load_df, lmp_df = self.sim_main.load_data()
        self.load_df = load_df
        self.lmp_df = self.sim_main.adjust_lmp_for_competition(
            lmp_df,
            n_agents=self.n_agents,
            elasticity=float(cfg.lmp_competition_elasticity),
            reference_agents=1,
        )

        state, action_space, market_model = self.sim_main.build_world(self.load_df, self.lmp_df)
        self.sim_main.apply_demand_scale_to_state(state, float(cfg.demand_scale))
        state.apply()

        self.state = state
        self.action_space = action_space
        self.market_model = market_model

        self.forecast_series = self.sim_main._load_ercot_forecast_series()

        fuel_mix_df = self.sim_main.load_fuel_mix()
        generators_df = self.sim_main.load_active_generators()
        self.segmented_load_df = self.sim_main.segment_load_by_fuel_and_capacity(
            self.load_df,
            fuel_mix_df,
            generators_df,
        )

        if not isinstance(self.state.raw_state_data, pd.DataFrame):
            raise RuntimeError("state.raw_state_data is not initialized")
        self.raw_state_data = self.state.raw_state_data

        safety_margin_hours = 48
        max_safe_start = len(self.raw_state_data) - self.state.window_size - safety_margin_hours
        self.episode_starts = list(range(0, max_safe_start + 1, self.state.step_hours))

        price_q = float(self.lmp_df[self.sim_main.PRICE_COL].abs().quantile(cfg.max_notional_q))
        self.max_notional = float(self.sim_main.MAX_BID_QUANTITY_MW * price_q)

    def _set_main_args(self) -> None:
        # shell.main functions are implemented against a module-level `args` object.
        self.sim_main.args = SimpleNamespace(
            demand_scale=float(self.cfg.demand_scale),
            load_cache_path=self.cfg.load_cache_path,
            verbose=False,
            lmp_competition_elasticity=float(self.cfg.lmp_competition_elasticity),
            supply_stack_price_noise_std=float(self.cfg.supply_stack_price_noise_std),
            max_notional_q=float(self.cfg.max_notional_q),
        )

    def evaluate_profile_once(self, joint_policies: Sequence[PolicyWrapper], seed: int) -> np.ndarray:
        if len(joint_policies) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} policies, got {len(joint_policies)}")

        rng = np.random.default_rng(seed)

        episodes = self.episode_starts[: self.cfg.n_eval_episodes]
        if not episodes:
            raise RuntimeError("No valid episodes available for empirical-game evaluation")

        cumulative = np.zeros(self.n_agents, dtype=float)

        for start_idx in episodes:
            self.state.episode_start = start_idx
            episode_start_ts = self.raw_state_data.index[start_idx]

            h_end = episode_start_ts + pd.Timedelta(hours=self.sim_main.FORECAST_HORIZON_HOURS)
            f_slice = self.forecast_series.loc[
                (self.forecast_series.index >= episode_start_ts) & (self.forecast_series.index <= h_end)
            ]
            forecast_df = f_slice.rename(self.sim_main.FORECAST_LOAD_COL).reset_index()
            self.sim_main.inject_epsisode_forecast(self.state, forecast_df)

            self.state.apply()
            obs_vec = self.state.reset(new_episode=False)
            state_key = state_vec_to_key(obs_vec)

            done = False
            step_counter = 0
            ep_rewards = np.zeros(self.n_agents, dtype=float)

            while not done:
                ts = self.state.timestamps[self.state.ptr]
                step_obs: Observation = {
                    "timestamp": ts,
                    "state_vec": obs_vec,
                    "state_key": state_key,
                }

                raw_actions = [int(policy.act(step_obs)) for policy in joint_policies]
                action_indices: list[int] = []
                for aidx in raw_actions:
                    proj_idx, _ = self.action_space.project_to_feasible(
                        int(aidx),
                        max_quantity=self.sim_main.MAX_BID_QUANTITY_MW,
                        max_notional=self.max_notional,
                    )
                    action_indices.append(int(proj_idx))

                delivery_time = ts + pd.Timedelta(hours=24)
                if delivery_time not in self.raw_state_data.index:
                    break

                delivery_row = self.raw_state_data.loc[delivery_time]
                historical_load_mw = delivery_row.get(self.sim_main.HIST_LOAD_COL, np.nan)
                if pd.isna(historical_load_mw):
                    break
                historical_load_mw = float(historical_load_mw)

                opponent_bids = self.sim_main.generate_opponent_bids_for_timestep(
                    delivery_time,
                    self.segmented_load_df,
                    self.lmp_df,
                    fuel_type_order=self.sim_main.SUPPLY_STACK_ORDER,
                    price_noise_std=float(self.cfg.supply_stack_price_noise_std),
                    rng=rng,
                )
                if not opponent_bids and historical_load_mw > 0:
                    break

                rl_bids = self.sim_main.extract_rl_bids_from_actions(action_indices, self.action_space)
                clearing_result = self.sim_main.clear_market_with_stack_competition(
                    rl_bids,
                    opponent_bids,
                    total_historical_load=historical_load_mw,
                    clearing_price=None,
                )

                clearing_price = float(clearing_result["clearing_price"])
                if clearing_price <= 0.0 and historical_load_mw > 1.0:
                    if clearing_result["total_cleared"] < historical_load_mw - 1e-6:
                        fallback = delivery_row.get(self.sim_main.PRICE_COL, np.nan)
                        if pd.isna(fallback):
                            valid_prices = self.lmp_df[self.sim_main.PRICE_COL].dropna()
                            fallback = float(valid_prices.iloc[-1]) if not valid_prices.empty else 1.0
                        clearing_price = max(float(fallback), 1.0)

                marginal_cost = float(self.market_model.params.marginal_cost)
                rl_cleared = clearing_result["rl_cleared"]
                for i in range(self.n_agents):
                    ep_rewards[i] += (clearing_price - marginal_cost) * float(rl_cleared[i])

                next_obs, _, done, _ = self.state.step()
                obs_vec = next_obs
                state_key = state_vec_to_key(next_obs)
                step_counter += 1

                if step_counter >= self.state.window_size:
                    done = True

            cumulative += ep_rewards

        return cumulative / float(len(episodes))


def _build_wrapped_policies(
    *,
    simulator: CapstoneReducedGameSimulator,
    policy_path: str,
    q_table_path: str | None,
    markup: float,
    quantile: float,
) -> tuple[dict[int, PolicyWrapper], dict[int, list[PolicyWrapper]], str]:
    saved_agents = load_saved_agents(policy_path=policy_path, q_table_path=q_table_path, fallback_action=0)

    if len(saved_agents) < simulator.n_agents:
        raise ValueError(
            f"Policy file has {len(saved_agents)} agents but n_agents={simulator.n_agents}."
        )

    final_wrappers: dict[int, PolicyWrapper] = {}
    baselines_by_agent: dict[int, list[PolicyWrapper]] = {}

    marginal_cost = float(simulator.market_model.params.marginal_cost)

    for agent_id in range(simulator.n_agents):
        saved_agent = saved_agents[agent_id]

        if not hasattr(saved_agent, "policy_map"):
            raise ValueError("Saved agent does not expose policy_map; expected SavedPolicyAgent structure.")

        learned_id = f"learned_agent_{agent_id}"
        final_wrappers[agent_id] = make_policy_wrapper_from_map(
            policy_id=learned_id,
            agent_id=agent_id,
            policy_map=saved_agent.policy_map,
            fallback_action=0,
            metadata={"policy_kind": "learned", "source": "policy.pkl"},
        )

        cost_plus = CostPlusMarkupPolicy(
            action_space=simulator.action_space,
            marginal_cost=marginal_cost,
            markup=float(markup),
            quantity_mw=float(simulator.sim_main.MAX_BID_QUANTITY_MW),
        )
        hist_q = HistoricalQuantilePolicy(
            action_space=simulator.action_space,
            lmp_df=simulator.lmp_df,
            quantile=float(quantile),
            quantity_mw=float(simulator.sim_main.MAX_BID_QUANTITY_MW),
        )

        baselines_by_agent[agent_id] = [
            make_policy_wrapper_from_callable(
                policy_id=f"cost_plus_agent_{agent_id}",
                agent_id=agent_id,
                act_fn=cost_plus.act,
                metadata={"policy_kind": "baseline", "baseline_name": "cost_plus"},
            ),
            make_policy_wrapper_from_callable(
                policy_id=f"hist_quantile_agent_{agent_id}",
                agent_id=agent_id,
                act_fn=hist_q.act,
                metadata={"policy_kind": "baseline", "baseline_name": "hist_quantile"},
            ),
        ]

    target_profile_id = "|".join(f"a{i}:{final_wrappers[i].policy_id}" for i in range(simulator.n_agents))
    return final_wrappers, baselines_by_agent, target_profile_id


def run_empirical_weak_acyclicity_test(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    GraphAnalysisResult,
]:
    sim_cfg = CapstoneSimulatorConfig(
        n_agents=int(args.n_agents),
        n_eval_episodes=int(args.eval_episodes_per_rollout),
        load_cache_path=str(args.load_cache_path),
        demand_scale=float(args.demand_scale),
        lmp_competition_elasticity=float(args.lmp_competition_elasticity),
        supply_stack_price_noise_std=float(args.supply_stack_price_noise_std),
        max_notional_q=float(args.max_notional_q),
    )
    simulator = CapstoneReducedGameSimulator(sim_cfg)

    final_wrappers, baselines_by_agent, target_profile_id = _build_wrapped_policies(
        simulator=simulator,
        policy_path=str(args.policy_path),
        q_table_path=str(args.q_table_path) if args.q_table_path else None,
        markup=float(args.baseline_markup),
        quantile=float(args.baseline_quantile),
    )

    library_options = LibraryBuildOptions(
        include_final_policy=True,
        include_baselines=bool(args.include_baselines),
        include_local_perturbations=bool(args.include_perturbations),
    )

    libraries = build_policy_libraries(
        final_policies=final_wrappers,
        action_space=simulator.action_space,
        baselines_by_agent=baselines_by_agent,
        options=library_options,
    )
    profile_space = build_profile_space(libraries)

    mc_cfg = MonteCarloConfig(
        n_rollouts=int(args.n_rollouts),
        base_seed=int(args.base_seed),
        confidence=float(args.confidence),
        use_common_random_numbers=bool(args.use_common_random_numbers),
        show_progress=bool(args.show_progress),
        profile_progress_every=int(args.profile_progress_every),
        rollout_progress_every=int(args.rollout_progress_every),
    )
    estimator = MonteCarloPayoffEstimator(simulator=simulator, config=mc_cfg)
    payoff_df, rollout_by_profile, _ = estimator.estimate_all_profiles(profile_space)

    br_cfg = BestResponseConfig(
        epsilon=float(args.epsilon),
        decision_rule=cast(DecisionRule, args.decision_rule),
        confidence=float(args.confidence),
    )
    deviation_df = deviation_table(
        profile_space=profile_space,
        rollout_by_profile=rollout_by_profile,
        br_config=br_cfg,
    )
    edge_df = best_reply_edges(deviation_df)
    gap_df = best_response_gaps(deviation_df)

    all_profile_ids = [p.profile_id for p in profile_space.profiles]
    sinks = sink_profiles(all_profile_ids=all_profile_ids, edge_df=edge_df)

    graph_result = analyze_weak_acyclicity(
        all_nodes=all_profile_ids,
        edge_df=edge_df,
        sinks=sinks,
    )

    sink_df = sink_quality_summary(
        sink_ids=sinks,
        deviation_df=deviation_df,
        gap_df=gap_df,
    )

    local_robust_df = local_robustness_around_profile(
        target_profile_id=target_profile_id,
        deviation_df=deviation_df,
    )

    if args.output_dir:
        export_tables(
            output_dir=str(args.output_dir),
            payoff_df=payoff_df,
            deviation_df=deviation_df,
            edge_df=edge_df,
            gap_df=gap_df,
            sink_df=sink_df,
            local_robustness_df=local_robust_df,
            graph_result=graph_result,
        )

    return payoff_df, deviation_df, edge_df, gap_df, sink_df, local_robust_df, graph_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Empirical weak-acyclicity test for a finite reduced game induced by the capstone simulator."
        )
    )

    p.add_argument("--policy_path", type=str, default="policy.pkl", help="Path to saved learned policy file.")
    p.add_argument("--q_table_path", type=str, default="q_table.pkl", help="Optional Q-table path for fallback.")

    p.add_argument("--n_agents", type=int, default=2, help="Number of RL agents in reduced game.")
    p.add_argument("--n_rollouts", type=int, default=16, help="Monte Carlo rollouts per joint profile.")
    p.add_argument(
        "--eval_episodes_per_rollout",
        type=int,
        default=5,
        help="Episodes averaged inside each rollout sample.",
    )
    p.add_argument("--base_seed", type=int, default=123, help="Base random seed for rollouts.")
    p.add_argument(
        "--use_common_random_numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use same rollout seed list across profiles for variance reduction.",
    )
    p.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress/ETA while evaluating reduced-game profiles.",
    )
    p.add_argument(
        "--profile_progress_every",
        type=int,
        default=1,
        help="Print a profile-level progress line every N completed profiles.",
    )
    p.add_argument(
        "--rollout_progress_every",
        type=int,
        default=0,
        help="If >0, also print rollout-level progress every N rollouts within each profile.",
    )

    p.add_argument("--epsilon", type=float, default=0.0, help="Approximate strict BR threshold.")
    p.add_argument(
        "--decision_rule",
        choices=["plain", "conservative"],
        default="plain",
        help="Best-reply decision: mean gain > epsilon or lower CI bound > epsilon.",
    )
    p.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI/LB computations.")

    p.add_argument(
        "--include_baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include baseline policies in each agent's reduced library.",
    )
    p.add_argument(
        "--include_perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include +/-1-bin local perturbations of learned policy.",
    )

    p.add_argument("--baseline_markup", type=float, default=10.0, help="Markup for cost-plus baseline.")
    p.add_argument("--baseline_quantile", type=float, default=0.7, help="Quantile for historical baseline.")

    p.add_argument("--load_cache_path", type=str, default="load_cache.csv")
    p.add_argument("--demand_scale", type=float, default=1.0)
    p.add_argument("--lmp_competition_elasticity", type=float, default=0.15)
    p.add_argument("--supply_stack_price_noise_std", type=float, default=2.0)
    p.add_argument("--max_notional_q", type=float, default=0.95)

    p.add_argument("--output_dir", type=str, default="Analysis/empirical_game")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payoff_df, deviation_df, edge_df, gap_df, sink_df, local_robust_df, graph_result = run_empirical_weak_acyclicity_test(args)

    print("Empirical reduced-game weak-acyclicity test complete")
    print(f"Profiles evaluated: {payoff_df['profile_id'].nunique()}")
    print(f"Best-reply edges: {len(edge_df)}")
    print(f"Sinks: {len(graph_result.sinks)}")
    print(f"Weakly acyclic (empirical, reduced game): {graph_result.weakly_acyclic}")
    if graph_result.problematic_nodes:
        print(f"Problematic nodes (no path to sink): {len(graph_result.problematic_nodes)}")
