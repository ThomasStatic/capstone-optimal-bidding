from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from shell.linear_approximator import Discretizer


@dataclass
class ActionSpace:
    """
    Discrete action space for bidding in the power market.
    Each action corresponds to a (price, quantity) pair, where both dimensions
    """
    price_disc: Discretizer       # fitted discretizer for bid price
    quantity_disc: Discretizer    # fitted discretizer for bid quantity

    def __post_init__(self) -> None:
        price_edges = self.price_disc.edges_
        qty_edges = self.quantity_disc.edges_
        if price_edges is None:
            raise ValueError("price_disc.edges_ is None; call fit(...) first.")
        if qty_edges is None:
            raise ValueError("quantity_disc.edges_ is None; call fit(...) first.")
        

        # Check that the discretizers look fitted and have edges_
        for name, disc in (("price_disc", self.price_disc),
                           ("quantity_disc", self.quantity_disc)):
            if not hasattr(disc, "edges_"):
                raise TypeError(f"{name} is missing 'edges_' attribute; expected a fitted discretizer.")
            if disc.edges_ is None:
                raise ValueError(f"{name}.edges_ is None; call fit(...) on the discretizer before using ActionSpace.")
            if len(disc.edges_) < 2:
                raise ValueError(f"{name}.edges_ must have at least 2 entries (one bin).")

        # Number of bins is implied by the edges: n_bins = len(edges) - 1
        self.n_price_bins: int = len(price_edges) - 1
        self.n_quantity_bins: int = len(qty_edges) - 1
        self.n_actions: int = self.n_price_bins * self.n_quantity_bins

        if self.n_price_bins <= 0 or self.n_quantity_bins <= 0:
            raise ValueError("Both price and quantity discretizers must define at least one bin.")

    def pair_to_index(self, price_bin: int, quantity_bin: int) -> int:
        """
        Convert a pair of bin indices (price_bin, quantity_bin) to a single
        discrete action index in [0, n_actions - 1].
        """
        if not (0 <= price_bin < self.n_price_bins):
            raise IndexError(
                f"price_bin {price_bin} out of range [0, {self.n_price_bins - 1}]."
            )
        if not (0 <= quantity_bin < self.n_quantity_bins):
            raise IndexError(
                f"quantity_bin {quantity_bin} out of range "
                f"[0, {self.n_quantity_bins - 1}]."
            )

        # Row-major flattening: price is the "row", quantity is the "column".
        return price_bin * self.n_quantity_bins + quantity_bin

    def index_to_pair(self, action_index: int) -> Tuple[int, int]:
        """
        Inverse of pair_to_index: convert a discrete action index back to
        (price_bin, quantity_bin).
        """
        if not (0 <= action_index < self.n_actions):
            raise IndexError(
                f"action_index {action_index} out of range [0, {self.n_actions - 1}]."
            )

        price_bin, quantity_bin = divmod(action_index, self.n_quantity_bins)
        return price_bin, quantity_bin

    def encode_from_values(self, price: float, quantity: float) -> int:
        """
        Convert a continuous (price, quantity) pair into a discrete action index.

        Uses the fitted bin edges from the discretizers and np.digitize to
        assign each value to a bin, then flattens (price_bin, quantity_bin)
        into a single action index.
        """

        price_edges = self.price_disc.edges_
        qty_edges = self.quantity_disc.edges_
        if price_edges is None:
            raise RuntimeError("price_disc.edges_ is None; call fit(...) first.")
        if qty_edges is None:
            raise RuntimeError("quantity_disc.edges_ is None; call fit(...) first.")

        # Map price to a price bin
        p_idx = np.digitize([price], price_edges)[0] - 1
        p_idx = int(np.clip(p_idx, 0, self.n_price_bins - 1))

        # Map quantity to a quantity bin
        q_idx = np.digitize([quantity], qty_edges)[0] - 1
        q_idx = int(np.clip(q_idx, 0, self.n_quantity_bins - 1))

        return self.pair_to_index(p_idx, q_idx)

    def decode_to_values(self, action_index: int) -> Tuple[float, float]:
        """
        Convert an action index back into representative (price, quantity)
        values defined by the bin edges.

        The representative value for a bin is taken as the centre of the bin:
            (left_edge + right_edge) / 2.
        """
        price_edges = self.price_disc.edges_
        qty_edges = self.quantity_disc.edges_
        if price_edges is None:
            raise RuntimeError("price_disc.edges_ is None; call fit(...) first.")
        if qty_edges is None:
            raise RuntimeError("quantity_disc.edges_ is None; call fit(...) first.")

        p_bin, q_bin = self.index_to_pair(action_index)

        # Price value = centre of its bin
        p_left = price_edges[p_bin]
        p_right = price_edges[p_bin + 1]
        price = 0.5 * (p_left + p_right)

        # Quantity value = centre of its bin
        q_left = qty_edges[q_bin]
        q_right = qty_edges[q_bin + 1]
        quantity = 0.5 * (q_left + q_right)

        return float(price), float(quantity)
    
    def project_to_feasible(
            self,
            action_index: int,
            *,
            max_quantity: float,
            max_notional: Optional[float] = None,
    ) -> Tuple[int, Dict[str, float | bool]]:
        """
        Takes a discrete action index, clamps it to risk limits, then re-encodes back to a discrete action index.

        Risk limits:
           - qty <= max_quantity
           - |price| * qty <= max_notional (if max_notional is not None)
        """
        orig_price, orig_qty = self.decode_to_values(action_index)

        new_price = float(orig_price)
        new_qty = float(np.clip(orig_qty, 0.0, max_quantity))

        orig_notional = float(abs(orig_price) * orig_qty)
        new_notional = float(abs(new_price) * new_qty)

        clipped = False

        if max_notional is not None and max_notional >= 0:
            if abs(new_price) > 1e-12 and (abs(new_price) * new_qty) > max_notional:
                # Reduce quantity to fit notional limit
                new_qty = float(max_notional / abs(new_price))
                new_qty = float(np.clip(new_qty, 0.0, max_quantity))
                clipped = True

        # If quantity clamp changed the quantity, mark as clipped
        if abs(new_qty - orig_qty) > 1e-12:
            clipped = True
        
        # Re-encode to nearest discrete action
        projected_index = self.encode_from_values(new_price, new_qty)

        # Decode to get final execution values
        exec_price, exec_qty = self.decode_to_values(projected_index)
        exec_notional = float(abs(exec_price) * exec_qty)

        info = {
            "clipped": clipped,
            "original_price": float(orig_price),
            "original_quantity": float(orig_qty),
            "original_notional": float(orig_notional),
            "executed_price": float(exec_price),
            "executed_quantity": float(exec_qty),
            "executed_notional": float(exec_notional),
            "max_quantity": float(max_quantity),
            "max_notional": float(max_notional) if max_notional is not None else float("nan"),
        }
        return projected_index, info
