from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class ActionSpace:
    """
    Discrete action space for bidding in the power market.

    Each action corresponds to a (price, quantity) pair, where both dimensions
    are discretized by separate 1D discretizers.

    We assume each discretizer object provides:
        - n_bins: int
        - to_index(value: float) -> int
        - from_index(index: int) -> float
    """

    price_disc: Any       # discretizer for bid price
    quantity_disc: Any    # discretizer for bid quantity

    def __post_init__(self) -> None:
        # Minimal sanity checks that the required attributes/methods exist.
        for name, disc in (("price_disc", self.price_disc), ("quantity_disc", self.quantity_disc)):
            if not hasattr(disc, "n_bins"):
                raise TypeError(f"{name} is missing 'n_bins' attribute.")
            if not hasattr(disc, "to_index"):
                raise TypeError(f"{name} is missing 'to_index(value: float) -> int' method.")
            if not hasattr(disc, "from_index"):
                raise TypeError(f"{name} is missing 'from_index(index: int) -> float' method.")

        self.n_price_bins: int = int(self.price_disc.n_bins)
        self.n_quantity_bins: int = int(self.quantity_disc.n_bins)
        self.n_actions: int = self.n_price_bins * self.n_quantity_bins

        if self.n_price_bins <= 0 or self.n_quantity_bins <= 0:
            raise ValueError("Both price_disc.n_bins and quantity_disc.n_bins must be positive.")

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

        The exact bucketing behaviour (bin boundaries, clipping, etc.) is
        determined entirely by the underlying discretizers.
        """
        p_bin = self.price_disc.to_index(price)
        q_bin = self.quantity_disc.to_index(quantity)
        return self.pair_to_index(p_bin, q_bin)

    def decode_to_values(self, action_index: int) -> Tuple[float, float]:
        """
        Convert an action index back into representative (price, quantity)
        values defined by the discretizers.
        """
        p_bin, q_bin = self.index_to_pair(action_index)
        price = self.price_disc.from_index(p_bin)
        quantity = self.quantity_disc.from_index(q_bin)
        return price, quantity
