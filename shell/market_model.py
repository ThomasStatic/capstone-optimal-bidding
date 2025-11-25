from dataclasses import dataclass
from typing import Tuple
import random


@dataclass
class MarketParams:
    """
    Settings for the market and the generator.

    min_price and max_price are passed in explicitly,
    typically taken from the price discretizer's edges.
    """
    marginal_cost: float        # cost per unit (e.g., $/MWh)
    price_noise_std: float      # std dev of noise around forecast price
    min_price: float            # lower bound for prices
    max_price: float            # upper bound for prices


class MarketModel:
    """
    This class:
      - decodes actions into (bid_price, bid_quantity),
      - samples a clearing price from a forecast,
      - decides how much quantity is cleared,
      - computes the reward (profit).
    """

    def __init__(self, action_space, params: MarketParams) -> None:
        self.action_space = action_space
        self.params = params

    def sample_clearing_price(self, forecast_price: float) -> float:
        """
        clearing_price = forecast_price + Gaussian noise,
        then clipped into [min_price, max_price].
        """
        noise = random.gauss(0.0, self.params.price_noise_std)
        price = forecast_price + noise
        price = max(self.params.min_price, min(self.params.max_price, price))
        return price

    def fill_fraction(self, bid_price: float, clearing_price: float,
                      eps: float = 1e-6) -> float:
        """
        Fraction of bid quantity that gets cleared.

        Logic:
        - If bid_price < clearing_price  -> 1.0 (full clear)
        - If bid_price > clearing_price  -> 0.0 (no clear)
        - If bid_price == clearing_price -> random fraction in [0, 1]
        """
        diff = bid_price - clearing_price

        # treat very small differences as equality (due to float noise)
        if abs(diff) < eps:
            return random.random()

        if bid_price < clearing_price:
            return 1.0  # cheaper than market -> fully dispatched
        else:
            return 0.0  # more expensive than market -> not dispatched

    def compute_reward(self, clearing_price: float,
                       cleared_quantity: float) -> float:
        """
        Reward = profit = (clearing_price - marginal_cost) * cleared_quantity.
        """
        c = self.params.marginal_cost
        profit = (clearing_price - c) * cleared_quantity
        return profit

    def clear_market_from_action(
        self,
        action_index: int,
        forecast_price: float,
    ) -> Tuple[float, float, float]:
        """
        Inputs:
          - action_index: discrete action chosen by the agent.
          - forecast_price: current forecast price from the state.

        Returns:
          - clearing_price (float)
          - cleared_quantity (float)
          - reward = profit (float)
        """
        # 1. decode (bid_price, bid_quantity) from your ActionSpace
        bid_price, bid_quantity = self.action_space.decode_to_values(action_index)

        # 2. sample a clearing price using the forecast
        clearing_price = self.sample_clearing_price(forecast_price)

        # 3. compute how much of the bid gets cleared
        f = self.fill_fraction(bid_price, clearing_price)
        cleared_quantity = f * bid_quantity

        # 4. compute the reward (profit)
        reward = self.compute_reward(clearing_price, cleared_quantity)

        return clearing_price, cleared_quantity, reward
