from dataclasses import dataclass
import math
from typing import Any, Dict, List, Sequence, Tuple
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

    def peek_reward_from_action(
        self,
        action_index: int,
        forecast_price: float,
    ) -> float:
        """Preserves RNG state for warm starting Q-values"""
        rng_state = random.getstate()
        try:
            _cp, _cq, reward = self.clear_market_from_action(action_index, forecast_price)
            return float(reward)
        finally:
            random.setstate(rng_state)
    
    def residual_share(
        self,
        P: float,
        *,
        rho_min: float,
        rho_max: float,
        k: float,
        p0: float,
    ) -> float:
        """
        rho(P) := rho_min + (rho_max - rho_min) / (1 + exp(k(P - p0)))
        """
        rho_lo = float(min(rho_min, rho_max))
        rho_hi = float(max(rho_min, rho_max))
        x = float(k) * (float(P) - float(p0))

        # Numerically stable logistic:
        # 1 / (1 + exp(x)) can overflow when x is large, so branch.
        if x >= 0:
            denom = 1.0 + math.exp(-x)
            sig = 1.0 / denom
        else:
            ex = math.exp(x)
            sig = ex / (1.0 + ex)

        # sig here is 1/(1+exp(-x)); we want 1/(1+exp(x)) which is (1 - sig)
        inv = 1.0 - sig  # = 1/(1+exp(x))

        rho = rho_lo + (rho_hi - rho_lo) * inv
        # Clamp to [0,1] because it's a share
        return float(max(0.0, min(1.0, rho)))

    def clear_market_multi_agent_residual(
        self,
        action_indices: Sequence[int],
        *,
        clearing_price: float,
        demand_mw: float,
        rho_min: float,
        rho_max: float,
        rho_k: float,
        rho_p0: float,
        tie_break_random: bool = True,
        eps: float = 1e-9,
    ) -> Dict[str, Any]:
        """
        Multi-agent clearing, pay-as-cleared, with residual demand:
            D_res(P) = D * rho(P)
        """
        n = len(action_indices)
        P = float(clearing_price)
        D = max(0.0, float(demand_mw))

        rho = self.residual_share(P, rho_min=rho_min, rho_max=rho_max, k=rho_k, p0=rho_p0)
        D_res = D * rho

        # Decode bids
        bids: List[Dict[str, float]] = []
        for i, aidx in enumerate(action_indices):
            bid_price, bid_qty = self.action_space.decode_to_values(int(aidx))
            bids.append({
                "agent_id": float(i),
                "bid_price": float(bid_price),
                "bid_qty": float(bid_qty),
                "action_index": float(aidx),
            })

        # Eligible if bid <= clearing price
        eligible = [b for b in bids if b["bid_price"] <= P + eps]

        # Merit order: lowest bid first, with optional random tie-breaking
        if tie_break_random:
            random.shuffle(eligible)
        eligible.sort(key=lambda b: b["bid_price"])

        cleared = [0.0] * n
        remaining = float(D_res)

        for b in eligible:
            if remaining <= eps:
                break
            i = int(b["agent_id"])
            take = min(float(b["bid_qty"]), remaining)
            cleared[i] = take
            remaining -= take

        rewards = [float(self.compute_reward(P, q)) for q in cleared]

        return {
            "clearing_price": P,
            "demand_mw": D,
            "rho": float(rho),
            "residual_demand_mw": float(D_res),
            "cleared_quantities": cleared,
            "rewards": rewards,
            "remaining_residual_demand": float(max(0.0, remaining)),
        }
