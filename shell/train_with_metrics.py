import matplotlib.pyplot as plt

from shell.api_controllers.market_loads_api import ISODemandController
from shell.market_model import MarketModel, MarketParams
from shell.tabular_q_agent import TabularQLearningAgent
from shell.rl_metrics_tabular import RLMetricsTrackerTabular
from shell.linear_approximator import PRICE_COL

import shell.main as main
from shell.main import (
    ISO,
    START_DATE,
    END_DATE,
    build_state_and_discretizers,
    load_data,
    make_action_space,
)

# Create and register a global args object so functions in shell.main
# that rely on module-level `args` (e.g., build_state_and_discretizers)
# work correctly when imported.
args = main.parse_args()
main.args = args

N_EPSIODES = args.n_episodes
MAX_STEPS_PER_EPISODE = 500


def train_with_metrics(show_plots: bool = True):
    # load_data() returns (load_df, lmp_df)
    load_df, lmp_df = load_data()

    state, _ = build_state_and_discretizers(load_df, lmp_df)
    action_space = make_action_space(lmp_df)

    market_params = MarketParams(
        marginal_cost=20.0,
        price_noise_std=5.0,
        min_price=float(action_space.price_disc.edges_[0]),
        max_price=float(action_space.price_disc.edges_[-1]),
    )
    market_model = MarketModel(action_space, market_params)

    agent = TabularQLearningAgent(num_actions=action_space.n_actions)

    metrics = RLMetricsTrackerTabular(num_actions=action_space.n_actions)

    # Debug buffers to inspect economic feasibility on this ISO.
    clearing_prices: list[float] = []
    bid_prices: list[float] = []
    cleared_quantities: list[float] = []
    rewards: list[float] = []

    for episode in range(N_EPSIODES):
        observation = state.reset(new_episode=(episode > 0))
        state_key = agent.state_to_key(observation)
        total_reward = 0.0

        max_steps = min(MAX_STEPS_PER_EPISODE, state.n_steps() - 1)
        episode_actions: list[int] = []

        for t in range(max_steps):
            action = agent.select_action(state_key)
            episode_actions.append(action)

            current_time = state.get_current_time()
            if state.raw_state_data is not None and current_time is not None:
                raw_current_row = state.raw_state_data.loc[current_time]
                forecast_price_val = raw_current_row[PRICE_COL]
                # Ensure we pass a scalar float into the market model.
                try:
                    forecast_price = float(forecast_price_val)
                except (TypeError, ValueError):
                    # If conversion fails (e.g., all NaNs), skip this step.
                    continue

                # Decode the bid associated with this action.
                bid_price, bid_qty = action_space.decode_to_values(action)

                clearing_price, cleared_qty, reward = market_model.clear_market_from_action(
                    action,
                    forecast_price,
                )

                # Per-step debug print to see economics.
                print(
                    f"t={t} | LMP={forecast_price:.2f} | "
                    f"bid_price={bid_price:.2f} | qty={bid_qty:.2f} | "
                    f"cleared={cleared_qty:.2f} | "
                    f"mc={market_params.marginal_cost:.2f} | "
                    f"reward={reward:.6f}"
                )

                clearing_prices.append(float(clearing_price))
                bid_prices.append(float(bid_price))
                cleared_quantities.append(float(cleared_qty))
                rewards.append(float(reward))

                next_observation, _, done, _ = state.step(action)
                next_state_key = None if done else agent.state_to_key(next_observation)

                if next_state_key is not None:
                    agent.update_q_table(
                        state_key, action, reward, next_state_key, done
                    )
                    state_key = next_state_key
                    total_reward += reward

                if done:
                    break

            agent.decay_epsilon()

        delta_q, delta_policy, action_counts = metrics.update(
            Q=agent.Q,
            episode_actions=episode_actions,
        )

        print(
            f"Episode {episode + 1}/{N_EPSIODES} | "
            f"steps: {t + 1} | "
            f"total reward: {total_reward:.2f} | "
            f"epsilon: {agent.epsilon:.3f}"
        )
        print(
            f"  [Metrics] ΔQ={delta_q}, "
            f"Δπ={delta_policy}, "
            f"action_counts={action_counts}"
        )

    print("Training with metrics complete.")

    # Basic diagnostics: is profit even achievable under this ISO config?
    if rewards:
        avg_reward = float(sum(rewards) / len(rewards))
        avg_clearing = float(sum(clearing_prices) / len(clearing_prices))
        avg_bid = float(sum(bid_prices) / len(bid_prices))
        avg_cleared_qty = float(sum(cleared_quantities) / len(cleared_quantities))
        positive_steps = sum(1 for r in rewards if r > 0)
        print("\n=== PJM profitability diagnostics ===")
        print(f"Steps observed: {len(rewards)}")
        print(f"Avg clearing price: {avg_clearing:.2f}")
        print(f"Avg bid price:      {avg_bid:.2f}")
        print(f"Avg cleared qty:    {avg_cleared_qty:.2f} MW")
        print(f"Avg reward/step:    {avg_reward:.6f}")
        print(f"Fraction steps with positive reward: {positive_steps/len(rewards):.4f}")

    if show_plots:
        plot_metrics(metrics, action_space)

    return agent, state, action_space, market_model, metrics


def plot_metrics(metrics: RLMetricsTrackerTabular, action_space):

    episodes_q = list(range(1, len(metrics.q_deltas) + 1))
    episodes_pi = list(range(1, len(metrics.policy_deltas) + 1))

    plt.figure(figsize=(14, 5))

    # delta Q plot
    plt.subplot(1, 3, 1)
    plt.plot(episodes_q, metrics.q_deltas, marker="o")
    plt.title("Q-value Convergence (ΔQ)")
    plt.xlabel("Episode")
    plt.ylabel("ΔQ (max |ΔQ(s,a)|)")

    # delta policy plot
    plt.subplot(1, 3, 2)
    plt.plot(episodes_pi, metrics.policy_deltas, marker="o")
    plt.title("Policy Convergence (Δπ)")
    plt.xlabel("Episode")
    plt.ylabel("Fraction of states changed")

    # action distribution plot
    plt.subplot(1, 3, 3)
    if metrics.action_histories:
        last_counts = metrics.action_histories[-1]
        actions = list(range(len(last_counts)))

        labels = [f"{p},{q}" for (p, q) in (action_space.index_to_pair(a) for a in actions)]

        plt.bar(actions, last_counts)
        plt.title("Action Distribution (last episode)")
        plt.xlabel("(price_bin, qty_bin)")
        plt.ylabel("Count")
        plt.xticks(actions, labels, rotation=90)
    else:
        plt.text(0.5, 0.5, "No actions recorded", ha="center", va="center")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_with_metrics()
